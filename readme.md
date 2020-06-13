ssh -NL localhost:16006:localhost:6006 root@140.115.70.199 -p 34524
ssh -NL localhost:16006:localhost:6006 140.115.70.212
rsync --progress --update -arvzh 
rsync --progress --ignore-existing -arvzh -e "ssh -p 34524" "root@140.115.70.199:/workplace/pclin/Deep-Learning-Final/predict/test/*" predict/test/
