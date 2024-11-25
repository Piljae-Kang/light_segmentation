# 🚀 Setup Instructions

Follow these steps to set up the project environment:

## Clone the Repository
Clone this repository to your local machine and navigate into the project directory:
```bash
git clone https://github.com/Piljae-Kang/light_segmentation.git
cd light_segmentation
```

## Create the Conda Environment
Use the provided environment.yml file to create a Conda environment with all required dependencies:
```bash
conda env create -f environment.yml
conda activate light_segmentation
```

# Run

## segmentation with 4 or 8 patterns
```bash
python segmentation_with_8patterns.py --root_path {} --material {} --phase_gap {} --output_path {}
Example : python segmentation_with_8patterns.py --root_path /home/piljae/Dataset/Hubitz/light_path_segmentation/8patterns --material metal_shaft --phase_gap 8
```

## segmentation with patterns
```bash
python segmentation_with_pattern.py --root_path {} --material {} --output_root_path {}
Example : python segmentation_with_pattern.py --root_path /home/piljae/Dataset/Hubitz/light_path_segmentation/scan_images --material metal_bell --output_root_path /home/piljae/Dropbox/hubitz/light_segmentation/segmentation_result
```
