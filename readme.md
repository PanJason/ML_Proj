# ML project: fracture detection

This is a project for the course Machine Learning. The purpose of this project is to detect fractures in chest bones.

## Method

## Dataset

## Additional Annotation

files: 
+ Additional_anno_train.json
+ Additional_anno_val.json

```json
{
  "bbox":
  {
    ...
  },
  "poly":
  {
    ...
  }
}
```

### Boundbox of chest
Item: "bbox".

Each one is a key-value pair.

Key: picture id.

Value: `[[left, top],[width, height]]`

### Outline and ribs
Item: "poly".

Each on is a key-value pair.

Key: picture id.

Value: a list of polylines.

`[(x1,y1),(x2,y2),...,(xn,yn)]`

The first polyline is the outline of chest bones. Marked clockwise, from bottom-left to top to bottom-right.

The other polylines is ribs. Each one of them started from the outline to spine. Arranged clockwise, from bottom-left to top to bottom-right. 


