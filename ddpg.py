from collections import deque
from os import path
import math
from copy import deepcopy
import random
import itertools
import functools

import torch
import torchvision
import torchvision.transforms.functional as TTF
import numpy as np

import model
import option
import data


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(
            m.weight.data, torch.nn.init.calculate_gain("relu"))
    elif classname.find('BatchNorm') != -1 or classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(
            m.weight.data, torch.nn.init.calculate_gain("sigmoid"))
        torch.nn.init.constant_(m.bias.data, 0)
    elif classname == "LSTMCell":
        for name, param in m._parameters.items():
            if 'bias' not in name:
                torch.nn.init.xavier_normal_(
                    param, torch.nn.init.calculate_gain("relu"))


class RibTracerActor(torch.nn.Module):
    def __init__(self, params):
        super(RibTracerActor, self).__init__()

        self.rnn = torch.nn.LSTMCell(2, params.DDPGHiddenSize)
        self.outNet = torch.nn.Sequential(
            torch.nn.Linear(params.DDPGHiddenSize + 50, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, 8),
            torch.nn.Sigmoid(),
            torch.nn.Linear(8, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, pos, image, states):
        h, c = self.rnn(pos, states)
        x = self.outNet(torch.cat((h, image), 1))
        return x, (h, c)


class RibTracerCritic(torch.nn.Module):
    def __init__(self, params):
        super(RibTracerCritic, self).__init__()
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(50, 16),
            torch.nn.Sigmoid()
        )
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(params.DDPGHiddenSize * 2, 32),
            torch.nn.Sigmoid()
        )
        self.linear3 = torch.nn.Sequential(
            torch.nn.Linear(36, 16),
            torch.nn.Sigmoid()
        )
        self.bilinear = torch.nn.Bilinear(16, 16, 8)
        self.predict = torch.nn.Sequential(
            torch.nn.Sigmoid(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, pos, state, actorState, act):
        x = self.linear1(state)
        y = torch.cat(actorState, 1)
        y = self.linear2(y)
        y = torch.cat((y, act, pos), 1)
        y = self.linear3(y)
        x = self.bilinear(x, y)
        return self.predict(x)


class RibTracerDDPG:
    def __init__(self, params):
        self.params = params

        self.buffer = deque()
        self.observer = model.makeRibTracerObserveNet(params)
        self.actor = RibTracerActor(params)
        self.critic = RibTracerCritic(params)

        self.actor_target = RibTracerActor(params)
        self.critic_target = RibTracerCritic(params)

        self.actor.apply(weights_init)
        self.critic.apply(weights_init)
        self.observer.load_state_dict(
            torch.load(
                path.join(params.model_path, "ribTracerObserver.pt")
            )
        )
        self.observer.eval()

        RibTracerDDPG.hardUpdate(self.actor, self.actor_target)
        RibTracerDDPG.hardUpdate(self.critic, self.critic_target)

        self.criticLoss = torch.nn.SmoothL1Loss()

        self.actorOptim = torch.optim.Adam(
            self.actor.parameters(), self.params.actorLearningRate)
        self.criticOptim = torch.optim.Adam(
            self.critic.parameters(), self.params.criticLearningRate)

        self.epsilon = 0.0
        self.resetState()

    def cuda(self):
        self.observer = self.observer.cuda()
        self.actor = self.actor.cuda()
        self.actor_target = self.actor_target.cuda()
        self.critic = self.critic.cuda()
        self.critic_target = self.critic_target.cuda()
        self.actorOptim = torch.optim.Adam(
            self.actor.parameters(), self.params.actorLearningRate)
        self.criticOptim = torch.optim.Adam(
            self.critic.parameters(), self.params.criticLearningRate)
        return self

    def play(self, img, poly, direction, training=False):
        self.resetState()
        pos = poly[0]
        pos = np.clip(pos, 0, self.params.imageSize)
        embeds = self.observe(img, pos)
        actorState = (torch.zeros([1, self.params.DDPGHiddenSize]),
                      torch.zeros([1, self.params.DDPGHiddenSize]))
        if len(poly) > 1:
            self.recordPoly(poly)
        if self.params.useGPU:
            actorState = (actorState[0].cuda(),
                          actorState[1].cuda())
        track = [deepcopy(pos)]
        total_reward = 0.0
        cnt = 0
        if training and len(self.buffer) >= self.params.warmUpSize:
            self.epsilon += self.params.DDPGepsilonDelta
        for i in range(self.params.maxSteps):
            tpos = torch.tensor(pos/self.params.imageSize, dtype=torch.float)
            if self.params.useGPU:
                tpos = tpos.cuda()
            cnt += 1
            with torch.no_grad():
                act, nextActorState = self.actor(
                    tpos.view([1, 2]), embeds, actorState)
                if training:
                    if len(self.buffer) <= self.params.warmUpSize:
                        act = self.randomDir()
                    else:
                        act += self.OU() * math.exp(-self.epsilon)
                        act = act.clamp(0.0, 1.0)

            if self.params.useGPU:
                step = act.cpu().numpy()[0]
            else:
                step = act.numpy()[0]
            step[1] = step[1] * 2.0 - 1.0
            step *= self.params.regionSize / 2.0 * self.params.traceStep
            if np.dot(step, direction) < 0:
                step = -step
            direction = step

            nextPos = pos + step
            nextPos = np.clip(nextPos, 0, self.params.imageSize)
            nextEmbeds = self.observe(img, nextPos)

            if len(poly) > 1:
                reward = self.calcReward(step, nextPos)
                failed = reward == self.params.failReward
                # finished = reward == self.params.finishReward
                total_reward += reward
                reward = torch.tensor(
                    [reward * self.params.rewardScale], dtype=torch.float)

                unfin = torch.tensor(
                    [i != self.params.maxSteps - 1 and not failed], dtype=torch.int)

                if self.params.useGPU:
                    reward = reward.cuda()
                    unfin = unfin.cuda()

                if training:
                    tnextPos = torch.tensor(
                        nextPos/self.params.imageSize, dtype=torch.float)
                    if self.params.useGPU:
                        tnextPos = tnextPos.cuda()
                    self.buffer.append(
                        (embeds, act, nextEmbeds, actorState,
                         nextActorState, reward, unfin,
                         tpos, tnextPos))
                    if len(self.buffer) > self.params.maxBuffer:
                        self.buffer.popleft()
                    if len(self.buffer) > self.params.warmUpSize:
                        for c in range(self.params.updateTimes):
                            self.update()

                if failed:
                    break
            embeds = nextEmbeds
            pos = nextPos
            actorState = nextActorState
            track.append(deepcopy(pos))
        # if training and len(self.buffer) > self.params.warmUpSize:
        #     for i in range(max(self.params.maxSteps - cnt, 0)):
        #         for c in range(self.params.updateTimes):
        #             self.update()
        return total_reward, track

    def update(self):
        state, act, nextState, actorState, nextActorState, reward, unfin, pos, nextPos = self.getBatch()

        with torch.no_grad():
            a, _ = self.actor_target(nextPos, nextState, nextActorState)
            nextQValue = self.critic_target(
                nextPos, nextState, nextActorState, a)

            targetQValue = reward + unfin * self.params.DDPGgamma * nextQValue

        self.critic.zero_grad()
        QValue = self.critic(pos, state, actorState, act)

        value_loss = self.criticLoss(QValue, targetQValue)
        value_loss.backward()

        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 5)
        self.criticOptim.step()

        self.actor.zero_grad()

        a, _ = self.actor(pos, state, actorState)
        policy_loss = -self.critic(pos, state, actorState, a)
        policy_loss = policy_loss.mean()
        policy_loss.backward()

        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 5)
        self.actorOptim.step()

        self.softUpdate(self.actor, self.actor_target)
        self.softUpdate(self.critic, self.critic_target)

        with torch.no_grad():
            self.total_value_loss += value_loss
            self.total_policy_loss += policy_loss
            self.update_cnt += 1

    def bufferSize(self):
        return len(self.buffer)

    def recordPoly(self, poly):
        self.poly = poly
        self.poly_lens = [np.linalg.norm(poly[i]-poly[i+1])
                          for i in range(len(poly)-1)]
        self.last_pos = poly[0]
        self.last_progress = 0.0
        self.last_dist = 0.0
        self.last_idx = 0

    def calcReward(self, step, pos):
        idx, dist, progress = self.findClosest(pos)
        speed = np.linalg.norm(step)

        dist_reward = self.last_dist - dist
        progess_reward = progress - self.last_progress

        self.last_pos = pos
        self.last_progress = progress
        self.last_dist = dist
        self.last_idx = idx

        if dist > self.params.regionSize:
            return self.params.failReward
        else:
            return dist_reward + progess_reward

    def findClosest(self, point):
        dists = [
            RibTracerDDPG.pointToLine(point, self.poly[i], self.poly[i+1])
            for i in range(len(self.poly)-1)
        ]
        idx = functools.reduce(
            lambda x, y: x if x[1][0] < y[1][0] else y, enumerate(dists))[0]
        dist, proj = dists[idx]
        progress = np.sum(self.poly_lens[:idx]) + proj
        return idx, dist, proj

    @staticmethod
    def pointProjection(P, A, B):
        AP = P - A
        AB = B - A
        return np.dot(AP, AB) / np.linalg(AB)

    @staticmethod
    def pointToLine(P, A, B):
        AP = P - A
        AB = B - A
        lenAB = np.linalg.norm(AB)
        proj = np.dot(AP, AB) / lenAB
        result = np.abs(np.cross(AP, AB)) / lenAB
        if proj < 0:
            result = np.linalg.norm(AP)
            proj = 0
        elif proj > lenAB:
            result = np.linalg.norm(P - B)
            proj = lenAB
        return [result, proj]

    def OU(self):
        self.OUState += self.params.OUtheta * self.params.OUdt * (-self.OUState) \
            + self.params.OUsigma * \
            torch.randn_like(self.OUState) * math.sqrt(self.params.OUdt)
        if self.params.useGPU:
            self.OUState = self.OUState.cuda()
        return self.OUState

    def resetState(self):
        self.OUState = torch.zeros([2])
        self.total_value_loss = torch.zeros(1)
        self.total_policy_loss = torch.zeros(1)
        self.update_cnt = 0

    def randomDir(self):
        x = torch.rand(1, 2)
        if self.params.useGPU:
            x = x.cuda()
        return x

    def observe(self, img, pos):
        x = TTF.crop(img, int(pos[1]), int(
            pos[0]), self.params.regionSize, self.params.regionSize)
        x = torchvision.transforms.ToTensor()(x)

        croped = torch.reshape(
            x, [1, 1, self.params.regionSize, self.params.regionSize])
        if self.params.useGPU:
            croped = croped.cuda()
        with torch.no_grad():
            result = self.observer(croped)
        return result

    def getBatch(self):
        # batch = [self.buffer.pop() for i in range(self.params.batchSize)]
        if self.params.batchSize <= len(self.buffer):
            batch = random.sample(self.buffer, self.params.batchSize)
        else:
            batch = random.sample(self.buffer, len(self.buffer))
        state = torch.stack([i[0] for i in batch], 1).squeeze()
        act = torch.stack([i[1] for i in batch], 1).squeeze()
        nextState = torch.stack([i[2] for i in batch], 1).squeeze()

        def stackState(states):
            h = torch.stack([i[0] for i in states], 1).squeeze()
            c = torch.stack([i[1] for i in states], 1).squeeze()
            return (h, c)
        actorState = stackState([i[3] for i in batch])
        nextActorState = stackState([i[4] for i in batch])
        reward = torch.stack([i[5] for i in batch])
        unfin = torch.stack([i[6] for i in batch])
        pos = torch.stack([i[7] for i in batch])
        nextPos = torch.stack([i[8] for i in batch])
        return state, act, nextState, actorState, nextActorState, reward, unfin, pos, nextPos

    def saveWeights(self):
        weights = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actorOptim": self.actorOptim.state_dict(),
            "criticOptim": self.criticOptim.state_dict()
        }
        torch.save(weights, path.join(
            self.params.model_path, "ribTracerDDPG.pt"))

    def loadWeights(self):
        w = torch.load(path.join(self.params.model_path, "ribTracerDDPG.pt"))
        self.actor.load_state_dict(w["actor"])
        self.actor_target.load_state_dict(w["actor_target"])
        self.critic.load_state_dict(w["critic"])
        self.critic_target.load_state_dict(w["critic_target"])
        self.actorOptim.load_state_dict(w["actorOptim"])
        self.criticOptim.load_state_dict(w["criticOptim"])

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    @staticmethod
    def hardUpdate(source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def softUpdate(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.params.DDPGtau) +
                param.data * self.params.DDPGtau
            )
