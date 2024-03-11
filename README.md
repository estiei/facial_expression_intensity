# AVI2024: intensity dataset

The code to extract timestamps with video fragments that contain human faces. Default duration ranges from 3 to 5 seconds. These extracted video segments are subsequently utilized for the assessment of facial expression intensity by human annotators.

## Extracting detected faces 

`face_search.py`: The script to extract timestamps.

## Required Arguments

- `--film_name`: Specifies the name of the film to process.

## Optional Arguments

- `--input_folder`: The directory where the input files are located. If not specified, it defaults to `./dataset/12movies_init_FaceReader_output`.

- `--output_folder`: The directory where the output files will be saved. If not specified, it defaults to `./data_intermediate/12movies_selected_frames`.

- `--lower_time_limit`: Sets the lower time limit (in seconds) for selecting frames. If not specified, it defaults to `3`.

- `--upper_time_limit`: Sets the upper time limit (in seconds) for selecting frames. If not specified, it defaults to `5`.

## Running the Script

To run the script with the minimum required arguments, use the following command:

`python script_name.py --film_name <your_movie_name>`

To customize the operation further, you can include any of the optional arguments, as shown in the example below:

`python script_name.py --film_name <your_movie_name> --input_folder <your_input_folder> --output_folder <./your_output_folder> --lower_time_limit 3 --upper_time_limit 5`

## Data

Data supporting the findings of this study will be made available upon the publication of this work.
