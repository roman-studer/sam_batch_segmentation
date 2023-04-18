from segment_anything import sam_model_registry


def get_sam_model(device):
    checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    path_checkpoint = f"./model/checkpoint/{checkpoint}"

    return sam_model_registry[model_type](path_checkpoint).to(device=device)



