
# the new config inherits the base configs to highlight the necessary modification
_base_ = './rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py'

# 1. dataset settings
dataset_type = 'DOTADataset'
classes = ('Fishing', 'Transport', 'Speedboat', 'Voilier', 'Military','Service')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/auguste/Bureau/MultiLabel-mmrotate/data/label',
        img_prefix='/home/auguste/Bureau/MultiLabel-mmrotate/data/Images'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/auguste/Bureau/MultiLabel-mmrotate/data/label',
        img_prefix='/home/auguste/Bureau/MultiLabel-mmrotate/data/Images'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/auguste/Bureau/MultiLabel-mmrotate/data/label',
        img_prefix='/home/auguste/Bureau/MultiLabel-mmrotate/data/Images'))

# 2. model settings
model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        # explicitly over-write all the `num_classes` field from default 15 to 5.
        num_classes=6))