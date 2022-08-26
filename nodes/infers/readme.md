

### summary ###
1. if the infer node works on the whole frame, let it derived from vp_primary_infer_node class.
2. if the infer node works on the small cropped images, let it derived from vp_secondary_infer_node class.
3. we can define multi derived classes to handle different types of dl models (detector/pose estimation/classification). also if they work on the same type of target AND with the same logic(for example, hava the same preprocess/postprocess), we can use a unique class to load different dl models(like resnet18 and resnet50).