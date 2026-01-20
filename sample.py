import paddle

state = paddle.load( "/mnt/d/Paderborn/StudyStuff/FinalYearProject/"
    "arcface_iresnet50_v1.0_pretrained/"
    "arcface_iresnet50_v1.0_pretrained.pdparams")

# Print first 30 parameter names
for i, k in enumerate(state.keys()):
    print(k)
    if i >= 30:
        break
