# Fast-EVA:  Fast Edge Video Analytics

## Demo Videos

- <img src="https://render.githubusercontent.com/render/math?math=\sigma">: actual frame processing rate on NCS2
- <img src="https://render.githubusercontent.com/render/math?math=\lambda">: the incoming video stream rate 
- <img src="https://render.githubusercontent.com/render/math?math=\mu">: frame processing rate at no frame dropping

### ETH-Sunnyday

| Original Video (<img src="https://render.githubusercontent.com/render/math?math=\lambda"> = 14 FPS) | Online Detection on one NCS2 <br/> (YOLOv3, <img src="https://render.githubusercontent.com/render/math?math=\sigma"> is set to <img src="https://render.githubusercontent.com/render/math?math=\mu">) <br/> slow detection processing rate <br/> <img src="https://render.githubusercontent.com/render/math?math=\sigma"> = 2.5, <img src="https://render.githubusercontent.com/render/math?math=\lambda"> = 14, <img src="https://render.githubusercontent.com/render/math?math=\mu"> = 2.5 |
|:---:|:---:|
| [![ETH-Sunnyday Original Video](https://j.gifs.com/MwM00O.gif)](https://youtu.be/BZZCMvbAKv0) | [![ETH-Sunnyday Slow Online Detection (YOLOv3, 1 NCS2, <img src="https://render.githubusercontent.com/render/math?math=\sigma"> is set to <img src="https://render.githubusercontent.com/render/math?math=\mu">)](https://j.gifs.com/p8EGGp.gif)](https://youtu.be/jFWfrZqeCUw) |

| Online Detection on one NCS2 <br/> (YOLOv3, <img src="https://render.githubusercontent.com/render/math?math=\sigma"> is set to <img src="https://render.githubusercontent.com/render/math?math=\lambda">) <br/> cause large random frame dropping <br/> <img src="https://render.githubusercontent.com/render/math?math=\sigma"> = 14, <img src="https://render.githubusercontent.com/render/math?math=\lambda"> = 14, <img src="https://render.githubusercontent.com/render/math?math=\mu"> = 2.5 | Online Detection on six NCS2s <br/> (YOLOv3, <img src="https://render.githubusercontent.com/render/math?math=\sigma"> is set to <img src="https://render.githubusercontent.com/render/math?math=\lambda">) <br/> significantly reduce random frame dropping <br/> <img src="https://render.githubusercontent.com/render/math?math=\sigma"> = 14, <img src="https://render.githubusercontent.com/render/math?math=\lambda"> = 14, <img src="https://render.githubusercontent.com/render/math?math=\mu"> = 14.8  |
|:---:|:---:|
| [![ETH-Sunnyday Online Detection (YOLOv3, 1 NCS2, <img src="https://render.githubusercontent.com/render/math?math=\sigma"> is set to <img src="https://render.githubusercontent.com/render/math?math=\lambda">)](https://j.gifs.com/oVDN2j.gif)](https://youtu.be/ZIks3oOGx8M) | [![ETH-Sunnyday Online Detection (YOLOv3, 6 NCS2, <img src="https://render.githubusercontent.com/render/math?math=\sigma"> is set to <img src="https://render.githubusercontent.com/render/math?math=\lambda">)](https://j.gifs.com/k8yJR5.gif)](https://youtu.be/0xu_d2RJ6YA) |
