## tips for BA node ##
- behaviour analysis（short as BA） works dependently on tracking, all BA nodes Must attached after track node in pipeline.
- all types of BA nodes support multi-channel, single instance of BA node supports multi channels as input.
- all BA nodes can be divided into 2 categories, one is instantaneous(like crossline) and another one is continuous(like stop, jam). `vp_ba_type` contains pair member for continuous BA such as `STOP` and `UNSTOP`, which means target enter BA status and leave BA status.
- there is no flag of BA inside `vp_frame_target` type, all nodes in pipeline need maintain it by themself if they want to know BA status for some task(like counter for crossline).