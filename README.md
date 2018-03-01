# baby_shazam

This is probably the weirdest project I've set myself to tackle. (What kinda people have baby cry audio files in their computer??) 

The project aims to create an automated classifier for baby cries. I stumbled upon this [paper](https://github.com/guigacarvalho/baby_shazam/raw/master/Detecting%20Pathologies%20from%20Infant%20Cry%20Applying%20Scaled%20Conjugate%20Gradient%20Neural%20Networks.pdf) and decided I should give it a try.

## Results

We've got around *72%* accuracy with this model, which is better than a random guess, but not ideal if you are a first-time parent with a screamming child. Here is an example of confusion matrix:

![confusion matrix](https://github.com/guigacarvalho/baby_shazam/blob/master/Result.png)

Better results would come with better audio processing (noise isolation), more classes and better ones (asphyxia and deafness are not exactly the go to application).

## Running the project
```bash
pip install -r requirements.txt
python process.py
```
