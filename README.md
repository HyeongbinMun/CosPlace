# SCA-Cosplays
본 프로젝트는 Cosplace matching을 이용해 로드뷰 사진을 매칭 구현한 프로젝트이다.
## Pipeline

## Project Architecture
```shell
.
├── README.md
├── data                        # data 디렉토리
├── docker                      # docker container 구축을 위해 필요한 설정파일
├── docker-compose.yml          # docker-compose 환경 설정 파일
├── docs                        # Readme 작성에 필요한 파일
├── models                      # Cosplace 관련 weight 및 모듈
├── eval.py                     # Cosplace 성능 평가를 위한 모듈
├── train.py                    # Cosplace 학습을 위한 모듈
├── test.py                     # Cosplace 결과 확인을 위한 모듈
├── requirements.txt            
├── scripts      
│   └── download.sh             # 성능평가 데이터셋 다운로드 스크립트
│   └── eval.sh                 # 성능평가 및 결과확인 스크립트
│   └── resize.sh               # 이미지 크기 재조정 스크립트
└── utils
    └── util.py                 # 모듈 실행을 위한 유틸리티
    └── common.py               # 모듈 실행을 위한 유틸리티
    └── img_resize.py           # 이미지 크기를 조절하기 위한 유틸리티
    └── recall.py               # 성능 평가를 위한 유틸리티

```

## Performance
### Method 별 성능
|Method| Recall@10   | Recall@5 | Recall@1 |
|------|----------|-----------|----------|
|Panorama| **1.00**    | 0.99      | 0.96     |
|cubemap| 0.85     | 0.82  | 0.66     |

## Installation
```shell
$ cd ${WORKSPACE}
$ git clone https://github.com/HyeongbinMun/SCA-Cosplays.git
$ cd SCA-Cosplays
$ vim docker-compose.yml
  # volume 경로, ports 수정
$ docker-compose up -d --build
$ docker attach ${CONTAINER NAME}
$ pip install -r requirements.txt
```
## Inference
Data preparation

- 파노라마 이미지(8000*4000)가 너무 크기 때문에 쿼리 이미지는 resize 필요(2000*1000)
- script에 있는 sample feature npy 파일은 (2000*1000)에 대한 feature vector임
- 이미지 사이즈를 맞추지 않으면 잘못된 검색을 할 수 있음
```shell
sh script/resize.sh
```
- resize.sh parameter 설명
  - origin_path : resize 대상 data
  - resize_path : save directory
  - img_type files : directory(폴더별로 들어가 있는 상태), files(이미지만 있는 상태)

visualize 결과
![image](https://user-images.githubusercontent.com/39808596/215614993-f8268b89-acff-4e3b-973c-74d53cfd4853.png)

## Evaluation
```shell
# 평가용 데이터 다운로드
sh scripts/download.sh
sh script/eval.sh
```
- eval.sh parameter 설명
  - db_dir : database directory
  - query_dirr : query directory
  - gt_dirr : ground truth directory
  - backbone : model
  - fc_output_dim : fc dimention
  - infer_batch_size : inference batch size
  - num_workers : num worker number
  - recall_value : recall count
  - resume_model : pretarin model path
  - feature_path : npy path
  - result_path : results path
  - result_txt_name : result txt name

다음과 같은 결과를 확인할 수 있다.
```shell
data load time : 28.79 sec
query search time : 7.27 sec
recall@10 : 100.00  # gt가 있는 경우에만 성능 표시
recall@5 : 100.00
recall@1 : 92.00
```
- 추가적으로 해당 recall 결과는 result path의 txt로 저장
- 해당 result에 따른 이미지는 result_path의 recall_img directory에서 확인 가능
