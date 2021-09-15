import torchreid

datamanager = torchreid.data.ImageDataManager(
    root='data',
    sources=['market1501','cuhk03','prid'],
    targets='sensereid',
    # height=256,
    # width=128,
    height=160,
    width=64,
    batch_size_train=24,
    batch_size_test=16,
    transforms=['random_flip', 'random_patch', 'color_jitter'],
    train_sampler='RandomIdentitySampler',
    combineall=True,
    cuhk03_classic_split=True
)

model = torchreid.models.build_model(
    name='hacnn',
    num_classes=datamanager.num_train_pids,
    loss='triplet',
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='multi_step',
    stepsize=[10, 15],
    gamma=0.5
)

engine = torchreid.engine.ImageTripletEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True,
    weight_t=0.5, 
    weight_x=1.0
)

# start_epoch = torchreid.utils.resume_from_checkpoint(
#     'log/pcb-new/model/model.pth.tar-10',
#     model,
#     optimizer
# )

engine.run(
    max_epoch=20,
    save_dir='log/hacnn-new',
    print_freq=50,
    eval_freq=5,
    dist_metric="cosine",
    # start_epoch=start_epoch
    # test_only=True
)