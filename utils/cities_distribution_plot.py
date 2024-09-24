import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Times new roman'

train_cities = [
    {'name': 'Barcelona', 'lon': 2.15, 'lat': 41.39},
    {'name': 'Cape Town', 'lon': 18.42, 'lat': -33.93},
    {'name': 'Los Angeles', 'lon': -118.24, 'lat': 34.05},
    {'name': 'Manila', 'lon': 121.00, 'lat': 14.62},
    {'name': 'Melbourne', 'lon': 144.97, 'lat': -37.81},
    {'name': 'Mexico City', 'lon': -99.14, 'lat': 19.43},
    {'name': 'New York', 'lon': -74.01, 'lat': 40.71},
    {'name': 'Paris', 'lon': 2.35, 'lat': 48.86},
    {'name': 'Santiago', 'lon': -70.66, 'lat': -33.45},
    {'name': 'Tokyo', 'lon': 139.76, 'lat': 35.68},
]

test_cities = [
    {'name': 'London', 'lon': -0.13, 'lat': 51.51},
    {'name': 'Taipei', 'lon': 121.50, 'lat': 25.05},
    {'name': 'Rio de Janeiro', 'lon': -43.18, 'lat': -22.90},
    {'name': 'Seattle', 'lon': -122.33, 'lat': 47.61},
    {'name': 'Sydney', 'lon': 151.21, 'lat': -33.87},
    {'name': 'Singapore', 'lon': 103.85, 'lat': 1.29},
]


fig = plt.figure(figsize=(10, 5), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())


ax.add_feature(cfeature.LAND, facecolor=(0.8, 0.8, 0.8))
# ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE, linewidth=0.2)


scat_train = ax.scatter([city['lon'] for city in train_cities],
                         [city['lat'] for city in train_cities],
                         c='red', s=30, transform=ccrs.Geodetic(), label='Train',)

scat_test = ax.scatter([city['lon'] for city in test_cities],
                        [city['lat'] for city in test_cities],
                        c='green', s=30, transform=ccrs.Geodetic(), label='Test')


ax.legend()

for city in train_cities + test_cities:
    if city['name'] in ['London', 'Melbourne']:
        text = ax.annotate(city['name'],
                    (city['lon'], city['lat']),
                    xytext=(-5, -5), textcoords='offset points',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    elif city['name'] in ['Barcelona']:
        text = ax.annotate(city['name'],
                    (city['lon'], city['lat']),
                    xytext=(5, -5), textcoords='offset points',
                    ha='left', va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    else:
        text = ax.annotate(city['name'],
                    (city['lon'], city['lat']),
                    xytext=(5, 5), textcoords='offset points',
                    ha='left', va='bottom',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))

ax.set_extent([-140, 180, -60, 75], crs=ccrs.PlateCarree())

plt.tight_layout()

plt.show()