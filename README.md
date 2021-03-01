<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# EVA:  Edge Video Analytics

## Demo Videos

- $\sigma$: actual frame processing rate on NCS2
- $\lambda$: the incoming video stream rate 
- $\mu$: frame processing rate at no frame dropping

### ETH-Sunnyday

| Original Video ($\lambda$ = 14 FPS) | Online Detection on one NCS2 <br/> (YOLOv3, $\sigma$ is set to $\mu$) <br/> $\sigma$ = 2.5, $\lambda$ = 14, $\mu$ = 2.5 |
|:---:|:---:|
| [![ETH-Sunnyday Original Video](https://j.gifs.com/MwM00O.gif)](https://youtu.be/BZZCMvbAKv0) | [![ETH-Sunnyday Slow Online Detection (YOLOv3, 1 NCS2, $\sigma$ is set to $\mu$)](https://j.gifs.com/p8EGGp.gif)](https://youtu.be/jFWfrZqeCUw) |

| Online Detection on one NCS2 <br/> (YOLOv3, $\sigma$ is set to $\lambda$) <br/> cause large random frame dropping <br/> $\sigma$ = 14, $\lambda$ = 14, $\mu$ = 2.5 | Online Detection on six Single NCS2s <br/> (YOLOv3, $\sigma$ is set to $\lambda$) <br/> $\sigma$ = 14, $\lambda$ = 14, $\mu$ = 14.8  |
|:---:|:---:|
| [![ETH-Sunnyday Online Detection (YOLOv3, 1 NCS2, $\sigma$ is set to $\lambda$)](https://j.gifs.com/oVDN2j.gif)](https://youtu.be/ZIks3oOGx8M) | [![ETH-Sunnyday Online Detection (YOLOv3, 6 NCS2, $\sigma$ is set to $\lambda$)](https://j.gifs.com/k8yJR5.gif)](https://youtu.be/0xu_d2RJ6YA) |
