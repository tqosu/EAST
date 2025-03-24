import torch
from collections import defaultdict

# Check if it's a checkpoint with a 'state_dict' key
# if 'state_dict' in checkpoint:
#     state_dict = checkpoint['state_dict']
# else:
#     state_dict = checkpoint

# # List the keys
# print("Keys in the .pth file:")
# for key in state_dict.keys():
#     print(key)

def main():
    # Your main logic goes here
    # pth_file_path = "exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter_2e-4_0.0002/split1/gpu2_id0/checkpoint/epoch_295.pth"
    
    epochs = [295, 235, 289, 269]
    mydict = defaultdict(list)
    for i, epoch in enumerate(epochs):
        pth_file_path = f"exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter_2e-4_0.0002/split{i+1}/gpu2_id0/checkpoint/epoch_{epoch}.pth"
        checkpoint = torch.load(pth_file_path, map_location="cpu")
    

        for j in range(24):
            # print(f'block {i}')
            w=checkpoint['state_dict'][f'module.backbone.model.backbone.blocks.{j}.adapter.up_proj.weight']
            g=checkpoint['state_dict'][f'module.backbone.model.backbone.blocks.{j}.adapter.gamma']
            b=checkpoint['state_dict'][f'module.backbone.model.backbone.blocks.{j}.adapter.up_proj.bias']
            weight0_l2 = torch.norm(w, p=2) #/ (w.numel() ** 0.5)
            bias0_l2 = torch.norm(b, p=2) #/ (b.numel() ** 0.5)
            set0_magnitude = weight0_l2 + bias0_l2
            print(f'block {j} set0_magnitude: {set0_magnitude*g}')
            mydict[j].append(set0_magnitude*g)
        # print(f'block {i} weight0_l2: {weight0_l2}')
        # print(f'block {i} bias0_l2: {bias0_l2}')
    for key, value in mydict.items():
        print(f'block {key} set0_magnitude: {sum(value)/len(value)}')

# This ensures main() runs only if the script is executed directly
if __name__ == "__main__":
    main()


# block 0 set0_magnitude: tensor([2.9796])
# block 1 set0_magnitude: tensor([2.9778])
# block 2 set0_magnitude: tensor([2.9438])
# block 3 set0_magnitude: tensor([2.9961])
# block 4 set0_magnitude: tensor([2.9440])
# block 5 set0_magnitude: tensor([3.0917])
# block 6 set0_magnitude: tensor([3.0973])
# block 7 set0_magnitude: tensor([3.1758])
# block 8 set0_magnitude: tensor([3.2300])
# block 9 set0_magnitude: tensor([3.2256])
# block 10 set0_magnitude: tensor([3.3763])
# block 11 set0_magnitude: tensor([3.4027])
# block 12 set0_magnitude: tensor([3.4621])
# block 13 set0_magnitude: tensor([3.4116])
# block 14 set0_magnitude: tensor([3.4995])
# block 15 set0_magnitude: tensor([3.3623])
# block 16 set0_magnitude: tensor([3.3189])
# block 17 set0_magnitude: tensor([3.2667])
# block 18 set0_magnitude: tensor([3.1354])
# block 19 set0_magnitude: tensor([3.1544])
# block 20 set0_magnitude: tensor([2.9508])
# block 21 set0_magnitude: tensor([2.9449])
# block 22 set0_magnitude: tensor([2.9098])
# block 23 set0_magnitude: tensor([2.8455])
