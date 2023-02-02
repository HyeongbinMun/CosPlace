# SCA-Cosplays
본 프로젝트는 Cosplace matching을 이용해 로드뷰 사진을 매칭 구현한 프로젝트이다.
## Pipeline
![image](https://user-images.githubusercontent.com/39808596/215600326-41533e0b-1493-4d01-88c6-6ea328b28dc3.png)

### Input
8000 * 4000 픽셀 크기의 파노라마 이미지 사용

하지만 기존 이미지의 크기가 너무 크기 때문에 샘플 데이터의 경우 2000 * 1000 픽셀 크기의 파노라마 이미지 사용
### Output
```python
result = {
   "query_image_1.jpg": [
        db_data/a.jpg,    # recall rank 1
        db_data/b.jpg,    # recall rank 2
        db_data/c.jpg,    # recall rank 3
        db_data/d.jpg,    # recall rank 4
        db_data/e.jpg     # recall rank 5
      ]
    }
```
### visualize 결과
![image](https://user-images.githubusercontent.com/39808596/215614993-f8268b89-acff-4e3b-973c-74d53cfd4853.png)

## Project Architecture
```shell
.
├── README.md
├── config                      
│   └── params.yaml             # parameter file
├── data                        # data 디렉토리
├── dataset                     
│   └── __init__.py             
│   └── test_dataset.py         # test dataset file
│   └── train_dataset.py        # train dataset file
├── docker                      # docker container 구축을 위해 필요한 설정파일
├── docker-compose.yml          # docker-compose 환경 설정 파일
├── docs                        # Readme 작성에 필요한 파일
├── models                      # Cosplace 관련 weight 및 모듈
├── parser.py                   # Cosplace 학습을 위한 파라미터 설정
├── eval.py                     # Cosplace 성능 평가를 위한 모듈
├── train.py                    # Cosplace 학습을 위한 모듈
├── test.py                     # Cosplace 결과 확인을 위한 모듈
├── requirements.txt            
├── scripts      
│   └── download.sh             # 성능평가 데이터셋 다운로드 스크립트
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
- 설치 방법은 아래 링크 참조

[HowToInstall.md](https://github.com/HyeongbinMun/SCA-Cosplays/tree/main/docs/docs/HowToInstall.md)

## Inference
- Inference 방법은 아래 링크 참조

[HowToInfer.md](https://github.com/HyeongbinMun/SCA-Cosplays/tree/main/docs/docs/HowToInfer.md)

## Evaluation
- Evaluation 방법은 아래 링크 참조

[HowToEval.md](https://github.com/HyeongbinMun/SCA-Cosplays/tree/main/docs/docs/HowToEval.md)




