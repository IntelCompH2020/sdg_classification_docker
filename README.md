# SDG Classification

This tool developed by Athena Reaserch Center classifies scientific literature into one or more Sustainable Development Goals (SDGs) as decribed by the [United Nations](https://sdgs.un.org/goals).

## Create Docker image

```
docker build --tag intelcomp_sdg -f ./dockerfile .
```

## Run image container

If there is already a container running

```
docker stop sdg_black_box_1
docker rm   sdg_black_box_1
``` 

Then we run an image container

```
docker container run -d -it --name sdg_black_box_1 -i intelcomp_sdg
```
