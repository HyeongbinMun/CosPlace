## Evaluation
 - Evaluation 진행 시 아래 양식과 같은 이미지 파일 필요(이름 앞에 ***id@*** 필요)
 - db directory
```
├── db_images                      
│   └── image_1.jpg         # 1@image_1.jpg의 db image
│   └── image_2.jpg         # 2@image_2.jpg의 db image
│   └── image_3.jpg         # 3@image_3.jpg의 db image
```
 - gt directory
```
├── gt_images                      
│   └── 1@image_1.jpg       # query new_image_1.jpg의 gt image
│   └── 2@image_2.jpg       # query new_image_2.jpg의 gt image
│   └── 3@image_3.jpg       # query new_image_3.jpg의 gt image
```
 - query directory
```
├── query_images                      
│   └── 1@new_image_1.jpg
│   └── 2@new_image_2.jpg
│   └── 3@new_image_3.jpg
```

 - 위 형식과 같은 디렉토리 구조가 있다면 recall value 확인 가능
```shell
python eval.py \
--query_path /workspace/data/sample/resize_queries \
--params /workspace/config/params.yaml
```
- eval.py 결과
```shell
data load time : 28.79 sec
query search time : 7.27 sec
query  result : [${DB DATA PATH}, ${DB DATA PATH}...] # recall이 가장 높은 값 출력
recall@10 : 100.00  # gt_path가 있는 경우에만 성능 표시
recall@5 : 100.00
recall@1 : 92.00
```

### hyperparameter
평가시 설정할 수 있는 파라미터는 cfg/config/params.yml 파일에서 설정할 수 있습니다.

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
