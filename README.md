<div style="text-align: center">

# 🌏🚶‍♂️🔍CV-Cities: Advancing Cross-View Geo-Localization in Global Cities

[//]: # (Paper: Under review  )
ArXiv: _🚧 coming soon..._
    
</div>

## Description

Cross-view geo-localization（CVGL） is beset with numerous difficulties and challenges, mainly due to the significant discrepancies in viewpoint, the intricacy of localization scenarios, and global localization needs. Given these challenges, we present a novel cross-view image geo-localization framework. The experimental results demonstrate that the proposed framework outperforms existing methods on multiple public datasets and self-built datasets. 
To improve the cross-view geo-localization performance of the framework on a global scale, we have built a novel global cross-view geo-localization dataset, CV-Cities. This dataset encompassing a diverse range of intricate scenarios. It serves as a challenging benchmark for cross-view geo-localization.

## CV-Cities: Global Cross-view Geo-localization Dataset
We collected 223,736 ground images and 223,736 satellite images with high-precision GPS coordinates of 16 typical cities in five continents. 
To download this dataset, you can click: [🤗CV-Cities](https://huggingface.co/datasets/gaoshuang98/CV-Cities) or [🤗CV-Cities (mirror)](https://hf-mirror.com/datasets/gaoshuang98/CV-Cities/tree/main).

### City distribution

<td style="text-align: center"><img src="/figures/distribution_map_of_cities.png" alt="City distribution" width="600"></td>

### Sample points distribution, 8 of 16 cities
<table style="text-align: center">
<tr>
<td style="text-align: center"><img src="/figures/capetown.png" alt="Capetown" width="100"></td>
<td style="text-align: center"><img src="/figures/london.png" alt="London" width="100"></td>
<td style="text-align: center"><img src="/figures/melbourne.png" alt="Melbourne" width="100"></td>
<td style="text-align: center"><img src="/figures/mexico.png" alt="mexico" width="100"></td>
</tr>
<tr>
<td style="text-align: center">Capetown, South Africa</td>
<td style="text-align: center">London, UK</td>
<td style="text-align: center">Melbourne, Australia</td>
<td style="text-align: center">Mexico city, Mexico</td>
</tr>
<tr>
<td style="text-align: center"><img src="/figures/newyork.png" alt="newyork" width="100"></td>
<td style="text-align: center"><img src="/figures/paris.png" alt="paris" width="100"></td>
<td style="text-align: center"><img src="/figures/rio.png" alt="rio" width="100"></td>
<td style="text-align: center"><img src="/figures/taipei.png" alt="Taipei" width="100"></td>
</tr>
<tr>
<td style="text-align: center">New York, USA</td>
<td style="text-align: center">Paris, France</td>
<td style="text-align: center">Rio, Brazil</td>
<td style="text-align: center">Taipei, China</td>
</tr>
</table>

### Different scenes
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
<td style="text-align: center" colspan="2">Water area</td>
<td style="text-align: center" colspan="2">Occlusion</td>
</tr>
<tr>
<td style="text-align: center"><img src="/figures/figure9-2-1.jpg" alt="ground image" width="150"></td>
<td style="text-align: center"><img src="/figures/figure9-2-2.jpg" alt="satellite image" width="75"></td>
<td style="text-align: center" rowspan="2" colspan="2">Other scenes...</td>
</tr>
<tr>
<td style="text-align: center" colspan="2">Season Change</td>
</tr>
</table>


### Yearly and monthly distribution
<table>
<tr>
<td style="text-align: center"><img src="/figures/figure3a.png" alt="Yearly distribution" width="200"></td>
<td style="text-align: center"><img src="/figures/figure3b.png" alt="monthly distribution" width="200"></td>
</tr>
</table>

## Framework
<td style="text-align: center"><img src="/figures/figure4.png" alt="Framework" width="500"></td>

## Precision distribution
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

## Model Zoo
_🚧 Under Construction 🛠️_

### Train the CVCities
```python
python train/train_cvcities.py
```

## Acknowledgments
This code is based on the amazing work of:
 - [DINOv2](https://github.com/facebookresearch/dinov2)
 - [Sample4Geo](https://github.com/Skyy93/Sample4Geo)

## Citation
_🚧 Under Construction 🛠_

