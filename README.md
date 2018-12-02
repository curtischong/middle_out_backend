# PyPiper

For Hack the North 2018 my team [Marcel O'neil](https://github.com/marceloneil), [Kevin Shen](https://github.com/kshen3778), [Michael Jiang](https://github.com/MichaelxhJiang), and I created PyPiper, a service that converts sketches of a scene into a complete animation.

### How it works

An animator would draw keyframes of a scene (ideally a couple of seconds apart) and then send them off to our servers. Next, we would take these frames and run it through a neural network to predict the frames in between these keyframes. Now that we have the outlines for each frame, we would then run a neural art mixer on the drawings to "colour in" the generated frames.

Bam! we have our animation!

Our program is called PyPiper because our neural network uses a revolutionary "Middle-Out" technique inspired by a paper from [Nvidia](https://arxiv.org/abs/1712.00080). We realized that we could predict intermediary frames by dropping every other frame from our input and use the dropped frames as the target variable. Using the two frames adjacent to the dropped frames, our U-net pipelines these two images to generate a new one. We heavily modified [Voxel-Flow's](https://github.com/liuziwei7/voxel-flow) Tensorflow architecture for our task. For those of you who are wondering, we are simply using an L1 loss as it has been shown to produce brilliant results.

### Steps for improvement
The number one thing we need to do is gather more varied data. Our current model trained on cut clips from the youtube channel  [TomSka](https://www.youtube.com/user/TomSka). Since their videos contained stick figures that were similar enough to keyframes, we thought that they would perfect for our use case. However, we discovered that most of the images were quite still so our network had trouble capturing the "motion" in the animation.

Interested in the front-end? Take a look at it [here](https://github.com/marceloneil/piepiper).
