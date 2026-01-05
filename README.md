## [PlutoNet: An efficient polyp segmentation network with modified partial decoder and decoder consistency training](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/htl2.12105)
by Tugberk Erol and Duygu Sarikaya

## Abstract
Deep learning models are used to minimize the number of polyps that goes unnoticed by the experts and to accurately segment the detected polyps during interventions. Although state-of-the-art models are proposed, it remains a challenge to define representations that are able to generalize well and that mediate between capturing low-level features and higher-level semantic details without being redundant. Another challenge with these models is that they are computation and memory intensive, which can pose a problem with real-time applications. To address these problems, PlutoNet is proposed for polyp segmentation which requires only 9 FLOPs and 2,626,537 parameters, less than 10% of the parameters required by its counterparts. With PlutoNet, a novel decoder consistency training approach is proposed that consists of a shared encoder, the modified partial decoder, which is a combination of the partial decoder and full-scale connections that capture salient features at different scales without redundancy, and the auxiliary decoder which focuses on higher-level semantic features. The modified partial decoder and the auxiliary decoder are trained with a combined loss to enforce consistency, which helps strengthen learned representations. Ablation studies and experiments are performed which show that PlutoNet performs significantly better than the state-of-the-art models, particularly on unseen datasets.

- MICCAI 2024 - AECAI
![sample](aecai2024.png)

## File tree
```
PlutoNet                           
├─images_article                          
│  ├─TestDataset                       
│  │  ├─CVC-300                 
│  │  │  ├─images
│  │  │  └─masks
│  │  ├─CVC-ClinicDB
│  │  │  ├─images
│  │  │  └─masks
│  │  ├─CVC-ColonDB
│  │  │  ├─images
│  │  │  └─masks
│  │  ├─ETIS-LaribPolypDB
│  │  │  ├─images
│  │  │  └─masks
│  │  └─Kvasir
│  │      ├─image
│  │      └─mask
│  └─train_images
│  └─train_masks   
│  └─val_images
   └─val_masks    
├─files
```
## Citation
- If you find this work is helpful, please cite our paper
```
@article{erol2024plutonet,
  title={PlutoNet: An efficient polyp segmentation network with modified partial decoder and decoder consistency training},
  author={Erol, Tugberk and Sarikaya, Duygu},
  journal={Healthcare Technology Letters},
  volume={11},
  number={6},
  pages={365--373},
  year={2024},
  publisher={Wiley Online Library}
}
```
