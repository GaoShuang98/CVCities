<div align="center">

# 🌏🚶‍♂️🔍CV-Cities: Advancing Cross-View Geo-Localization in Global Cities
[![Official Repo](http://img.shields.io/badge/official_repo-CVCities-007ACC.svg)](https://github.com/GaoShuang98/CVCities)
[![Paper](http://img.shields.io/badge/paper-IEEE_JSTARS-B31B1B.svg)](https://doi.org/10.1109/JSTARS.2024.3502160)
[![Paper](http://img.shields.io/badge/paper-arXiv.2411.12431-B31B1B.svg)](https://arxiv.org/abs/2411.12431)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cv-cities-advancing-cross-view-geo/visual-place-recognition-on-cv-cities)](https://paperswithcode.com/sota/visual-place-recognition-on-cv-cities?p=cv-cities-advancing-cross-view-geo)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cv-cities-advancing-cross-view-geo/image-based-localization-on-cvact)](https://paperswithcode.com/sota/image-based-localization-on-cvact?p=cv-cities-advancing-cross-view-geo)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cv-cities-advancing-cross-view-geo/image-based-localization-on-cvusa-1)](https://paperswithcode.com/sota/image-based-localization-on-cvusa-1?p=cv-cities-advancing-cross-view-geo)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cv-cities-advancing-cross-view-geo/image-based-localization-on-vigor-cross-area)](https://paperswithcode.com/sota/image-based-localization-on-vigor-cross-area?p=cv-cities-advancing-cross-view-geo)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cv-cities-advancing-cross-view-geo/image-based-localization-on-vigor-same-area)](https://paperswithcode.com/sota/image-based-localization-on-vigor-same-area?p=cv-cities-advancing-cross-view-geo)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cv-cities-advancing-cross-view-geo/drone-view-target-localization-on-university-1)](https://paperswithcode.com/sota/drone-view-target-localization-on-university-1?p=cv-cities-advancing-cross-view-geo)

</div>

## Description 📜

Cross-view geo-localization（CVGL） is beset with numerous difficulties and challenges, mainly due to the significant discrepancies in viewpoint, the intricacy of localization scenarios, and global localization needs. Given these challenges, we present a novel cross-view image geo-localization framework. The experimental results demonstrate that the proposed framework outperforms existing methods on multiple public datasets and self-built datasets. 
To improve the cross-view geo-localization performance of the framework on a global scale, we have built a novel global cross-view geo-localization dataset, CV-Cities. This dataset encompassing a diverse range of intricate scenarios. It serves as a challenging benchmark for cross-view geo-localization.

## CV-Cities: Global Cross-view Geo-localization Dataset 💾
We collected 223,736 ground images and 223,736 satellite images with high-precision GPS coordinates of 16 typical cities in five continents. 
To download this dataset, you can click: [🤗CV-Cities](https://huggingface.co/datasets/gaoshuang98/CV-Cities) or [🤗CV-Cities (mirror)](https://hf-mirror.com/datasets/gaoshuang98/CV-Cities/tree/main).

### City distribution 📊

<td style="text-align: center"><img src="/figures/distribution_map_of_cities.png" alt="City distribution" width="600"></td>

### Sample points and monthly distribution of 16 cities 📍
<table style="text-align: center">
<tr>
<td style="text-align: center"><img src="/figures/captown-month.png" alt="Capetown" width="100"></td>
<td style="text-align: center"><img src="/figures/london-month.png" alt="London" width="100"></td>
<td style="text-align: center"><img src="/figures/melbourne-month.png" alt="Melbourne" width="100"></td>
<td style="text-align: center"><img src="/figures/mexico-month.png" alt="mexico" width="100"></td>
<td rowspan="8"><img src="/figures/monthly-scalebar.png" alt="Colorbar" width="40"> <br>Colorbar</td>
</tr>
<tr>
<td style="text-align: center">Capetown, South Africa</td>
<td style="text-align: center">London, UK</td>
<td style="text-align: center">Melbourne, Australia</td>
<td style="text-align: center">Mexico city, Mexico</td>
</tr>
<tr>
<td style="text-align: center"><img src="/figures/newyork-month.png" alt="newyork" width="100"></td>
<td style="text-align: center"><img src="/figures/paris-month.png" alt="paris" width="100"></td>
<td style="text-align: center"><img src="/figures/rio-month.png" alt="rio" width="100"></td>
<td style="text-align: center"><img src="/figures/taipei-month.png" alt="Taipei" width="100"></td>
</tr>
<tr>
<td style="text-align: center">New York, USA</td>
<td style="text-align: center">Paris, France</td>
<td style="text-align: center">Rio, Brazil</td>
<td style="text-align: center">Taipei, China</td>
</tr>
<tr>
<td style="text-align: center"><img src="/figures/losangeles-month.png" alt="Losangeles" width="100"></td>
<td style="text-align: center"><img src="/figures/maynila-month.png" alt="Maynila" width="100"></td>
<td style="text-align: center"><img src="/figures/santiago-month.png" alt="Santiago" width="100"></td>
<td style="text-align: center"><img src="/figures/sydney-month.png" alt="Sydney" width="100"></td>
</tr>
<tr>
<td style="text-align: center">Losangeles, USA</td>
<td style="text-align: center">Manila, Philipine</td>
<td style="text-align: center">Santiago, Chile</td>
<td style="text-align: center">Sydney, Australia</td>
</tr>
<tr>
<td style="text-align: center"><img src="/figures/seattle-month.png" alt="Seattle" width="100"></td>
<td style="text-align: center"><img src="/figures/singapore-month.png" alt="Singapore" width="100"></td>
<td style="text-align: center"><img src="/figures/barcelona-month.png" alt="Barcelona" width="100"></td>
<td style="text-align: center"><img src="/figures/tokyo-month.png" alt="Tokyo" width="100"></td>
</tr>
<tr>
<td style="text-align: center">Seattle, USA</td>
<td style="text-align: center">Singapore</td>
<td style="text-align: center">Barcelona, Span</td>
<td style="text-align: center">Tokyo, Japan</td>
</tr>
</table>

### Different scenes 🏞️
<table>
<tr>
<td style="text-align: center"><img src="/figures/figure2-1-1.jpg" alt="ground image" width="150"></td>
<td style="text-align: center"><img src="/figures/figure2-1-2.jpg" alt="satellite image" width="75"></td>
<td style="text-align: center"><img src="/figures/figure2-1-3.jpg" alt="ground image" width="150"></td>
<td style="text-align: center"><img src="/figures/figure2-1-4.jpg" alt="satellite image" width="75"></td>
</tr>
<tr>
<td style="text-align: center" colspan="2">City scene</td>
<td style="text-align: center" colspan="2">Nature scene</td>
</tr>
<tr>
<td style="text-align: center"><img src="/figures/figure2-2-1.jpg" alt="ground image" width="150"></td>
<td style="text-align: center"><img src="/figures/figure2-2-2.jpg" alt="satellite image" width="75"></td>
<td style="text-align: center"><img src="/figures/figure2-2-3.jpg" alt="ground image" width="150"></td>
<td style="text-align: center"><img src="/figures/figure2-2-4.jpg" alt="satellite image" width="75"></td>
</tr>
<tr>
<td style="text-align: center" colspan="2">Water scene</td>
<td style="text-align: center" colspan="2">Occlusion</td>
</tr>

[//]: # (<tr>)
[//]: # (<td style="text-align: center"><img src="/figures/figure9-2-1.jpg" alt="ground image" width="150"></td>)
[//]: # (<td style="text-align: center"><img src="/figures/figure9-2-2.jpg" alt="satellite image" width="75"></td>)
[//]: # (<td style="text-align: center" rowspan="2" colspan="2">Other scenes...</td>)
[//]: # (</tr>)
[//]: # (<tr>)
[//]: # (<td style="text-align: center" colspan="2">Season Change</td>)
[//]: # (</tr>)

</table>


### Scenes, yearly and monthly distribution 📊
<table>
<tr>
<td style="text-align: center"><img src="/figures/figure3a.png" alt="Scenes distribution" width="200"></td>
<td style="text-align: center"><img src="/figures/figure3b.png" alt="Yearly distribution" width="200"></td>
<td style="text-align: center"><img src="/figures/figure3c.png" alt="monthly distribution" width="200"></td>
</tr>
</table>

## Framework 🖇️
<td style="text-align: center"><img src="/figures/figure4.png" alt="Framework" width="500"></td>

## Precision distribution 🚿
<table style="text-align: center">
<tr>
<td style="text-align: center"><img src="/figures/precision_london100.jpg" alt="London" width="150"></td>
<td style="text-align: center"><img src="/figures/precision_rio100.jpg" alt="Rio" width="150"></td>
<td style="text-align: center"><img src="/figures/precision_seattle100.jpg" alt="seattle" width="150"></td>
</tr>
<tr>
<td style="text-align: center">London, UK</td>
<td style="text-align: center">Rio, Brazil</td>
<td style="text-align: center">Seattle, USA</td>

</tr>
<tr>
<td style="text-align: center"><img src="/figures/precision_sinapore100.jpg" alt="Singapore" width="150"></td>
<td style="text-align: center"><img src="/figures/precision_sydney100.jpg" alt="sydney" width="150"></td>
<td style="text-align: center"><img src="/figures/precision_taipei100.jpg" alt="taipei" width="150"></td>
</tr>
<tr>
<td style="text-align: center">Singapore</td>
<td style="text-align: center">Sydney, Australia</td>
<td style="text-align: center">Taipei, China</td>
</tr>
</table>

## Model Zoo 📦
_🚧 Under Construction_

### Train the CVCities 🚂
```python
python train/train_cvcities.py
```

## Acknowledgments 🧭
This code is based on the amazing work of:
 - [DINOv2](https://github.com/facebookresearch/dinov2)
 - [Sample4Geo](https://github.com/Skyy93/Sample4Geo)
 - [QGIS](https://github.com/qgis/QGIS)

## Citation✅
```
  @ARTICLE{huangCVCities2024,
  author={Huang, Gaoshuang and Zhou, Yang and Zhao, Luying and Gan, Wenjian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={CV-Cities: Advancing Cross-View Geo-Localization in Global Cities}, 
  year={2025},
  volume={18},
  number={},
  pages={1592-1606},
  keywords={Urban areas;Satellite images;Accuracy;Feature extraction;Training;Location awareness;Buildings;Manuals;Benchmark testing;Transformers;Cross-view geo-localization (CVGL);dataset;global cities;image retrieval;visual place recognition},
  doi={10.1109/JSTARS.2024.3502160}}
```

