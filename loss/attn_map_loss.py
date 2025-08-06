import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import math
import torch.nn.functional as F
from Myutils.visualize2 import visualize_tensor


def find_token_indices(text_list, text_token, entity_lists=None):
    """
    text_list: List[str], batch of texts
    entity_list_list: List[List[str]], 每个文本对应的实体列表
    tokenizer: 支持 batch 的 tokenizer

    返回：
        List[List[List[int]]]  batch_size  x num_entities  x token_indices
    """
    if entity_lists is None:
        entity_lists = []
        for text in text_list:
            entities = text.replace(',', '').strip().split()
            entity_lists.append(entities)
    # 统一 tokenize，带偏移，batch 返
    offsets = text_token["offset_mapping"]  # shape: (batch_size, seq_len, 2)

    batch_token_indices = []
    for batch_idx, (text, entity_list) in enumerate(zip(text_list, entity_lists)):
        offset = offsets[batch_idx]  # shape: (seq_len, 2)

        # 1. 提前构造每个实体的起止位置（字符级）
        entity_spans = []
        for ent in entity_list:
            ent = ent.strip()
            if not ent:
                entity_spans.append(None)
                continue

            ent_start = text.find(ent)
            if ent_start == -1:
                entity_spans.append(None)
            else:
                entity_spans.append((ent_start, ent_start + len(ent)))

        # 2. 初始化每个实体对应的 token 索引列表
        matched_indices = [[] for _ in entity_spans]

        # 3. 只遍历一次 offset
        for token_idx, (start, end) in enumerate(offset):
            s, e = start.item(), end.item()
            if s == e:
                continue  # 跳过无效 token（如 CLS）

            for ent_idx, span in enumerate(entity_spans):
                if span is None:
                    continue
                ent_start, ent_end = span
                if e > ent_start and s < ent_end:
                    matched_indices[ent_idx].append(token_idx)

        batch_token_indices.append(matched_indices)

    return batch_token_indices


class attention_map_loss(nn.Module):
    def __init__(self,resolution=64):
        super(attention_map_loss, self).__init__()
        self.resolution=resolution
        self.image_transforms= transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.UpSample=transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR)
    def get_label_tensor(self,label,device=None):
        mask_tensor_list=[]
        for file in label:
            map_tensor=[self.image_transforms(img=Image.open(i).convert('L')) for i in file]
            if len(map_tensor)==0:
                mask_tensor_list.append(torch.tensor([]))
            else:
                mask_tensor_list.append(torch.cat(map_tensor,dim=0).to(device=device) if device is not None else torch.cat(map_tensor,dim=0))

        return mask_tensor_list



    def forward(self,label_list,attention_map_list):
        label_list=self.get_label_tensor(label_list,device=attention_map_list[0][0][0].device)
        count_list=[len(i)for i in label_list]
        loss_list=[]
        for crossDown_list in attention_map_list:
            for Transformer_list in crossDown_list:
                attn_map=Transformer_list[0]
                b,head,n,t=attn_map.shape
                attn_map=attn_map.mean(dim=1, keepdim=False).permute(0,2,1).view(b,-1,int(math.sqrt(n)),int(math.sqrt(n)))
                for c,am,l in zip(count_list,attn_map,label_list):
                    if c==0:
                        continue
                    l = F.softmax(l.view(c, -1), dim=-1).view(c,self.resolution,self.resolution)
                    # l=torch.where(l<0, torch.zeros_like(l), l)
                    am=am[:c,:,:]
                    am=self.UpSample(am)
                    # am = F.softmax(am.view(c, -1), dim=1).view(c, self.resolution, self.resolution)
                    loss_list.append(F.l1_loss(l, am, reduction='mean'))

        loss = torch.stack(loss_list, dim=0).mean(dim=0)
        return loss

class attention_map_loss_with_first_token(nn.Module):
    def __init__(self,resolution=64):
        super(attention_map_loss_with_first_token, self).__init__()
        self.resolution=resolution
        self.image_transforms= transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.UpSample=transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR)
    def get_label_tensor(self,label,device=None):
        mask_tensor_list=[]
        for file in label:
            map_tensor=[self.image_transforms(img=Image.open(i).convert('L')) for i in file]
            if len(map_tensor)==0:
                mask_tensor_list.append(torch.tensor([]))
            else:
                mask_tensor_list.append(torch.cat(map_tensor,dim=0).to(device=device) if device is not None else torch.cat(map_tensor,dim=0))

        return mask_tensor_list



    def forward(self,label_list,attention_map_list):
        label_list=self.get_label_tensor(label_list,device=attention_map_list[0][0][0].device)
        count_list=[len(i)for i in label_list]
        loss_list=[]
        for crossDown_list in attention_map_list:
            for Transformer_list in crossDown_list:
                attn_map=Transformer_list[0]
                b,head,n,t=attn_map.shape
                attn_map=attn_map.mean(dim=1, keepdim=False).permute(0,2,1).view(b,-1,int(math.sqrt(n)),int(math.sqrt(n)))
                for c,am,l in zip(count_list,attn_map,label_list):
                    if c==0:
                        continue
                    l = F.softmax(l.view(c, -1), dim=-1).view(c,self.resolution,self.resolution)
                    # l=torch.where(l<0, torch.zeros_like(l), l)
                    am=am[1:c+1,:,:]
                    am=self.UpSample(am)
                    # am = F.softmax(am.view(c, -1), dim=1).view(c, self.resolution, self.resolution)
                    loss_list.append(F.l1_loss(l, am, reduction='mean'))

        loss = torch.stack(loss_list, dim=0).mean(dim=0)
        return loss

class attention_map_loss_with_first_token_and_comma(nn.Module):
    def __init__(self, resolution=64):
        super().__init__()
        self.resolution = resolution
        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.UpSample = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR)

    def get_label_tensor(self, label, device=None):
        mask_tensor_list = []
        for file in label:
            file=sorted(file)
            map_tensor = [self.image_transforms(img=Image.open(i).convert('L')) for i in file]
            if len(map_tensor) == 0:
                mask_tensor_list.append(torch.tensor([]))
            else:
                mask_tensor_list.append(torch.cat(map_tensor, dim=0).to(device=device) if device is not None else torch.cat(map_tensor, dim=0))
        return mask_tensor_list

    def forward(self, label_list, attention_map_list, offset_mapping_list, text,text_token):
        """
        label_list: list of list of mask file paths (per sample)
        attention_map_list: list of attention maps from model (per layer/block)
        offset_mapping_list: list of offset mappings per sample, shape [num_tokens, 2]
        token_words: list of tokens (strings) per sample, e.g. ["car", ",", "person", ",", "curve", ",", "bollard", ",", "cone"]
        """

        label_list = self.get_label_tensor(label_list, device=attention_map_list[0][0][0][0].device)
        batch_size = len(label_list)
        loss_list = []

        token_indices=find_token_indices(text,text_token)


        for crossDown_list in attention_map_list:
            for Transformer_list in crossDown_list:
                attn_maps = Transformer_list[0]  # shape (b, head, n, t),因为这一层的list只有一个元素,所以直接取0
                b, head, n, t = attn_maps.shape
                attn_maps = attn_maps.mean(dim=1).permute(0, 2, 1).view(b, -1, int(math.sqrt(n)), int(math.sqrt(n))) # (b,token_num,h,w)

                for i in range(batch_size):
                    masks = label_list[i]  # shape (num_words, H, W), num_words=5
                    en, h, w = masks.shape
                    attn_map=attn_maps[i]
                    _, a_h, a_w = attn_map.shape
                    token_index=token_indices[i]
                    if len(token_index) == 0:
                        attn_map = torch.full_like(attn_map[0], 1/(h*w ),device=attn_map.device).unsqueeze(0)  # 保持维度一致
                    else:
                        attn_map = torch.stack([
                            attn_map[ids].mean(0) for ids in token_index
                        ])

                    attn_map = self.UpSample(attn_map)  # 调整到mask尺寸

                    # 3. 计算loss：mask和对应单词的attn_map做L1
                    masks = masks.to(attn_map.device).float()
                    masks = masks.view(en, -1)  # [B, N, 4096]
                    masks = torch.softmax(masks, dim=-1)  # 在每个 mask 的空间维度上 softmax
                    masks = masks.view(-1,h,w)
                    # mask shape: (num_words, H, W)
                    loss = F.l1_loss(masks, attn_map, reduction='mean')
                    loss_list.append(loss)

        return torch.stack(loss_list).mean()


if __name__ == '__main__':
    label=torch.rand((8,3,256,256))
