# LMMS Application for Word Sense Disambiguation

This repository is a fork of ["Language Modelling Makes Sense"](https://github.com/danlou/LMMS) for word-sense disambiguation (WSD). This repository provides the optimized code as a flask-application for finding the WSD of word(s) in a given sentence, utilizing pytorch BERT model and pre-trained word sense contexual embeddings.

Key features added:

- Development of end-to-end WSD API to get the detailed information from a sentence and it's words given as input. 
- BERT model weights conversion from tensorflow to pytorch
- Excluding burdensome process (client/server) architecture [bert-as-service](https://github.com/hanxiao/bert-as-service) to retrieve the pretrained BERT embeddings
- Optimization of code to get the WSD API output in less than 2 seconds on CPU (upto three words in a single request)

## Installation

 ``` bash
 
$ cd lmms_app
$ pip3 install -r requirements.txt
```

### Download pretrained SENSE EMBEDDINGS 

Manual Link: [.npz (0.3GB)](https://drive.google.com/uc?id=1kuwkTkSBz5Gv9CB_hfaBh1DQyC2ffKq0&export=download) 

Terminal:
``` bash
$ pip3 install gdown
$ gdown https://drive.google.com/uc?id=1kuwkTkSBz5Gv9CB_hfaBh1DQyC2ffKq0&export=download 
```

### Download pytorch pretrained BERT (large-cased) model files

This pytorch bert model weights are converted from this original model provided in the paper: [cased_L-24_H-1024_A-16](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)
Manual Link: [pytorch-bert-model](https://drive.google.com/u/0/uc?id=1NQ_3cGw1qWvc7tGwPlRHixzyjRP7lvgd&export=download)

Terminal:
``` bash
$ gdown https://drive.google.com/uc?id=1kuwkTkSBz5Gv9CB_hfaBh1DQyC2ffKq0&export=download
$ unzip bert_torch_model.zip
```

### Run the Application

Run the following command to deploy the flask application:

``` bash
$ python3 WSD_updated.py
```

### Invoke the API

Once the API is up and running, invoke the API through the following curl command:

``` bash
$ curl -X POST -d '{"sentence":"you were right that turning right was a better way", "word": ["right",
"turning"]}' http://127.0.0.1:5000/synset_processing -H "Content-Type: application/json" -w 'Total: %{time_total}s\n'
```

### Output

Sample output:

```bash
{
  "bert_WSD": [
    {
      "definition": "free from error; especially conforming to fact or truth", 
      "offset": 631391, 
      "synset": "Synset('correct.a.01')", 
      "synset_key": "right%3:00:02::", 
      "word": "right"
    }, 
    {
      "definition": "toward or on the right; also used figuratively", 
      "offset": 387828, 
      "synset": "Synset('right.r.04')", 
      "synset_key": "right%4:02:03::", 
      "word": "right"
    }, 
    {
      "definition": "a turn toward the side of the body that is on the north when the person is facing east", 
      "offset": 351168, 
      "synset": "Synset('left.n.05')", 
      "synset_key": "left%1:04:00::", 
      "word": "turning"
    }
  ]
}
Total: 1.363022s
```
