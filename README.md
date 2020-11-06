## **H&E-*ERBB2***

## Deep Learning Identifies Morphological Features in Breast Cancer Predictive of Cancer *ERBB2* Status and Trastuzumab Treatment Efficacy

___

![](misc/view.png)

### Sample data and models
📦 [Download link (~3.3Gb)](https://www.dropbox.com/sh/mi0hixo1l18j128/AADDz2jX9JSRbkciGFdm7rBDa?dl=1)

```
he-erbb2-github-data
 ├── tissue-samples
 │    ├── [ sample 1 ]
 │    ├──     ...
 │    └── [ sample n ]
 └── models
      └── Her2
           ├── fold-1.pth
           ├── fold-2.pth
           ├── fold-3.pth
           ├── fold-4.pth
           └── fold-5.pth
    
```

### Inference
Attach to the docker container and run:
```
python inference.py -c configs/inference-config.json
```
The script generates a `.csv` file with predicted scores.


### Running docker image
- Clone the Repository
- CD to the docker folder `cd docker`
- Edit "DATA_VOLM" path in `init-docker.sh` - point to the data you downloaded through the linkd, e.g. `/your/path/he-erbb2-github-data/`. Other variables can remain default.
- Build an image: `./build-docker.sh`
- Run a container: `./run-docker.sh`
  - Containers run in an interactive mode
  - <kbd>control</kbd> + <kbd>p</kbd>, <kbd>control</kbd> + <kbd>q</kbd> will turn interactive mode into daemon mode, i.e. detach from a container without stopping it
  - Reattach container: `docker attach [name]`
  - `Warning:` <kbd>control</kbd> + <kbd>a</kbd> <kbd>d</kbd> – detach from TMUX session when inside a container. This will kill the container!