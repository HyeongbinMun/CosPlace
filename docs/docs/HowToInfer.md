## Inference
 - db에 대한 첫 feature vector 추출 시 오랜 시간이 걸림
   - cfg/config/params.yml의 feature path에 npy file이 없을 경우 feature path에 npy file 생성
   - 파노라마 약 7만장의 경우 하루 정도 소요됨
 - db feature npy file 생성 시 기존 크기(8000,4000)으로 뽑을 건지 (2000,1000)으로 뽑을 건지 선택 필요
   - sample data의 경우 (2000,1000)으로 npy file 생성 -> inference image도 (2000,1000)으로 찾을 수 있음
   - 기존 이미지 외의 (2000,1000) image 사용 경우 아래의 코드를 통해 resize 필요
```python
python resize.sh
# script content
--origin_path /workspace/data/sample/test_query \       # origin path
--resize_path /workspace/data/sample/resize_queries \   # result path
--img_type files                                        # files or directory(아래 구조 참고 후 선택)
```
 - files 구조
 ```python
├── images
│   └── image_1.jpg
│   └── image_2.jpg
    ...
```
- directory 구조
 ```python
├── images
│   └── 1923
│       └── image_1.jpg
│   └── 1954
│       └── image_2.jpg
    ...
```
 - Simple python code
```python
from test import Cosplace

image_path = 'test.png'

# load model
model = Cosplace()

# image inference
result = model.inference_by_images(image_path)
```

### hyperparameter
추론시 설정할 수 있는 파라미터는 cfg/config/params.yml 파일에서 설정할 수 있습니다.

```yaml
model:
  data:
    db: /nfs_shared/STR_Data/roadview/cosplace/panorama_resize/1796    # db 경로
    gt: /workspace/data/sample/test_gt                                 # gt 경로
    feature: /workspace/models/feature/anyang_2019_panorama.npy        # feature 경로
    save: /workspace/results                                           # save 경로
    save_name: result.txt                                              # save txt 이름
  cosplace:
    model_name: cosplace101-512                                        # model 이름
    model_path: /workspace/models/weights/cosplace_resnet101_512.pth   # model 경로
    backbone: resnet101                                                # model backbone
    device: cuda                                                       # gpu : cuda, cpu : cpu
    fc_output_dim: 512                                                 # dimention number
    batch_size: 1                                                      # batch size 크기
    num_workers: 2                                                     # num worker 크기
    recall_value: 10                                                   # recall 개수
    seed: 0                                                            # seed number
```
