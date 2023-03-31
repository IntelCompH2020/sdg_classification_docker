# SDG Classification

This tool developed by Athena Reaserch Center classifies scientific literature into one or more Sustainable Development Goals (SDGs) as decribed by the [United Nations](https://sdgs.un.org/goals).

## Create Docker image

```
docker build --tag intelcomp_sdg -f ./dockerfile .
```

## Run image container as a demo

If there is already a container running remove it

```
docker stop sdg_black_box_1
docker rm   sdg_black_box_1
``` 

Then we run an image container 

```
docker container run -d -it --name sdg_black_box_1 -i intelcomp_sdg
```

Check whether the container is running

```
docker container ls --all
``` 

and Collect the output of the classifier
 
 ```
 docker cp sdg_black_box_1:app/resources/data/test_output.txt ./
```
 

## Run image container in production

You have to mount the input directory and the output directory

```
-v input_directory:directory_inside_docker"
-v output_directory:directory_outside_docker"
``` 

Example:

 ```
 docker run \
-v /home/dpappas/input_data:/input_files \
-v /home/dpappas/output_data:/output_files \
-i intelcomp_sdg \
batch_classifier.py \
--data_path=/input_files/test_input.txt \
--out_path=/output_files/test_output.txt

```
 



