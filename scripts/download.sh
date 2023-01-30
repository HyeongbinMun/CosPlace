echo "Download start cosplace model weights"
mkdir -p /workspace/models/weights/
wget ftp://mldisk.sogang.ac.kr/ganpan_matching/cosplace/resnet101_512.pth -O /workspace/models/weights/cosplace_resnet101_512.pth
echo "Download start anyang feature npy file"
mkdir -p /workspace/models/feature/
wget ftp://mldisk.sogang.ac.kr/ganpan_matching/cosplace/anyang_2019.npy -O /workspace/models/feature/anyang_2019.npy
wget ftp://mldisk.sogang.ac.kr/ganpan_matching/cosplace/anyang_2019_panorama.npy -O /workspace/models/feature/anyang_2019_panorama.npy
echo "Download start anyang sample panorama data"
mkdir -p /workspace/data/sample/
wget ftp://mldisk.sogang.ac.kr/ganpan_matching/cosplace/anyang_sample.zip -O /workspace/data/sample/anyang_sample.zip