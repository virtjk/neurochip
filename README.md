# neurochip
Converting chiptune and MIDI music into live arrangement mockups using a machine learning model.

## Intro

This technique involves feeding a chiptune or low-end MIDI rendering of a sketch into Stable Diffusion, and getting a dreamy "recorded version" back. The [Riffusion](https://github.com/riffusion/riffusion) system creates brand new music via spectrogram reconstruction. Normally, denoising is used to ensure the generated patterns are distinct from the base input image. By decreasing this value and using a clean sound source like chip-based synths, the model will produce a note-for-note arrangement that sounds roughly like a live cover.

This project exists to collect my utility scripts, guides, and/or observations in one place. Some basic familiarity with ML terminology and some command line proficiency are assumed, but I'll try to explain as much as I understand at any given point.

## Examples

First, some early tests using my own tracks; no retraining of the model with my prior work, just the MEL spectrogram of one song, and a prompt. The fidelity should be expected to improve as new models or datasets are incorporated, or upscaling tricks found. Beyond the stereo panning of paired generations, no reverb or other signal processing effects were used.

- [Input: Boccherini - Minuet](https://github.com/virtjk/neurochip/blob/main/examples/bocc-orig.wav?raw=true) Nintendo "chiptune" game music
- [Output: Boccherini - Minuet](https://github.com/virtjk/neurochip/blob/main/examples/bocc-out.wav?raw=true) Processed version
- [A folk tune](https://github.com/virtjk/neurochip/blob/main/examples/final_note.mp3?raw=true) - NES chiptune source, like the Minuet...

Crossfades, to show what the original (i.e. input) sounds like, followed by the "neural mix":

- [FM Synth - OPNA (YM-2608)](https://github.com/virtjk/neurochip/blob/main/examples/lostatlantis_crossfade.mp3?raw=true) - crossfading comparison between the original chip version and the "arrange album"
- [Amiga Soundtracker ST-01 Samples](https://github.com/virtjk/neurochip/blob/main/examples/keepshreddin_crossfade.mp3?raw=true) - crossfading comparison between the original 4-channel S3M tracker file and the "studio cut", and back
- [Nintendo NES](https://github.com/virtjk/neurochip/blob/main/examples/incidentzero_crossfade.mp3?raw=true) progressive metal NES album... now on tour.
- [Windows GM.DLS MIDI](https://github.com/virtjk/neurochip/blob/main/examples/silentprince.mp3?raw=true) This time it starts with the processed version, and ends up fading back to the raw input!


## How?!

[Riffusion](https://github.com/riffusion/riffusion) is a spectrogram-based music generation model. Its primary use is making music from scratch, based on a text prompt. Try it out at [Riffusion.com](https://riffusion.com)!

It can be run locally, and there are extensions available for image generation tools such as [Automatic1111's web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). If you are a VST plugin user, there is already a [Riffusion VST](https://github.com/mklingen/RiffusionVST) which could save you the time of rendering out mixes, especially if you want to try processing individual instrument stems. 

The main Riffusion package comes with a standalone inference server with its own API. It also comes with two apps that do almost everything you need to generate and process music: The Riffusion.com app itself, and the Streamlit "Riffusion Playground" app. This is currently my preferred method.

The app responds to changes and starts re-generating the output data automatically, so it's very fast and easy to experiment with it. If you would like to get it running locally, just make sure to follow the guide in the Riffusion Readme closely, to avoid frustration. 

## The Setup

If you just want a quick high-level summary of everything that's involved here:

1. Write a chiptune or simple-waveform synth track using [OpenMPT](https://openmpt.org/) or [Furnace](https://github.com/tildearrow/furnace) or Garageband or any other music tool. [Soundfont](https://en.wikipedia.org/wiki/SoundFont)-style music works fine, or a Sound Canvas, etc.
2. Render the song as a mono 44100 Hz / 16-bit .Wav (the format used internally by the model, although conversion is automated through ffmpeg).
4. Install [Riffusion](https://github.com/riffusion/riffusion).
5. Run the included Streamlit app and connect to it in your browser.
6. Click "Audio to Audio".
7. Turn the Denoising setting down enough to avoid changing the notes, so it's now a cover instead of a new composition.
8. Formulate a prompt in the style of your choosing, and use the Guidance parameter to tell the computer how much it should believe your prompt.
9. Render the entire song instead of a 5- or 20-second snippet, and save the output
10. (optional) tweak the parameters and render a second time for a slightly altered version to use for stereo doubling.



## A Quick Aesthetic Guide

Make sure Riffusion is properly installed, as per [the guide on their GitHub page](https://github.com/riffusion/riffusion)!

![instructions1](https://user-images.githubusercontent.com/14255864/218354185-73316dba-a21f-451d-9ce2-f3a6654ba925.png)

Once you have gone through the trials of Conda configuration and reached this screen in your web browser, click [Audio-to-Audio.](https://github.com/riffusion/riffusion) In your excitement, don't forget to look at the other features in the playground, like automatic instrument stem separation!

### Notes for Composers

This method seems to respond especially well to emotive midi playing that uses mod wheel, volume swells, or pitch bends. If you play synth vibrato, the "live" guitar will pick up on it, leading into and out of it naturally. It adds legato-sample style detail to the transitions between individual notes to give fast passages a more natural sound, without needing any key-switch articulation cues. It can interpret even plain beeps expressively if you dial up the Denoising. But the more effort you put into the details of the sketch, the lower you can crank both the Denoising and Guidance (CFG) and still hear a plausibly live performance.

As with other models in the news, it can be "confidently wrong" and place things like valve clicks, 3-armed drum hits, mysterious group shouts, and "piano fret squeaks" in arbitrary places. Some of that can be minimized with parameters and prompting. To me, it's part of the style.

Once you have a song scaffolding or entire chip track rendered, you can drag the wav or mp3 into the browser window, and drop it on the Upload slot. After that, you can begin filling out your prompt and algorithm settings.

## Settings

![instructions_shreddin2](https://user-images.githubusercontent.com/14255864/218354363-9e262153-f60e-4dca-aae9-e924c42d16f6.png)

**Tips on Prompts**

AI prompt crafting is an emerging art! Each stack of models will have its own labeling idiosyncrasies and a resulting "house style". Here, I recommend an earnest description of your song in its final form, as a caption or blurb on a fan wiki. I'm not opposed to large pedantically detailed lists, but I try to avoid adding specifics unless the inference engine fails to infer something I'm implying. I find being able to trust it to get the point is part of the thrill.

Either way, the process of narrowing down the right phrasing for your prompt, along with finding the "sweet spot" in the Guidance and Denoise settings on each track, will be where you spend the bulk of your effort, even for a seasoned composer who knows the "keeper" take when they hear it. This is a process of maximal cherry-picking... it can take dozens of refinements or new approaches before you capture the right vibe, OR it could be perfect on the first try. 

I can't say with any scientific rigor that my hyping up of the band via prompt really works as well as I feel it does, but... without it they do sound subjectively less enthusiastic about the slap bass or drum fills. There must be something to it.

**Seed**

The integer used to create a noise pattern to refine into something matching your description of the input. (ow my brain) Changing this by even one digit can affect the entire song's interpretation.

But changing this for the sake of variety can quickly become a slot machine-type addictive pull. Don't rely on more than occasional re-rolls. It can help to have a consistent pattern to chisel into shape. The seed itself seems to matter less as you reach a consistent prompt that produces lots of nearly ideal moods.

**Denoising**
The "Ship of Theseus" slider. This determines how much of your input track actually remains. The higher the value, the more "brand new" the output is. Going much past 0.5 with this will result in an unrecognizable jumble.

At the other extreme, with a value under 0.15 not much, if anything, will change in the timbre of the instruments. It's pretty common to just get a grungy filtered chiptune back at first. Then, you raise it until it starts changing the notes on you.

In my experience, the most useful range for the neurochip vibe is somewhere between 0.2 and 0.4 - Around 0.36 or so is where it starts chickening out on solos. But every song has a different "comfort zone" where it's still a note-for-note cover with new sounds.

## Sidebar Settings
![instructions_shreddin1](https://user-images.githubusercontent.com/14255864/218354374-c92ca892-214b-4fd8-abbb-77f88c119564.png)

**Steps per Sample**

Each denoising algorithm has different aesthetic results, but I start with the defaults for that and the Steps per Sample setting. It's usually the last thing I mess with. Here's what changing it does:

- [Original input (NES)](https://github.com/virtjk/neurochip/blob/main/examples/curio_original.mp3?raw=true)
- [5 Steps](https://github.com/virtjk/neurochip/blob/main/examples/curio_5steps.mp3?raw=true)  -- Wow, it's sometimes a violin! And some unwanted bass drums. 
- [25 Steps](https://github.com/virtjk/neurochip/blob/main/examples/curio_25steps.mp3?raw=true) -- Wow, it's ... almost a violin! But more detailed now
- [50 Steps](https://github.com/virtjk/neurochip/blob/main/examples/curio_50steps.mp3?raw=true) -- It's still not quite a violin, and we're reaching diminishing returns, but the articulations and dynamics of each note are sharper.

A very low (and thus fast to generate) value does produce some pretty amazing "intentional jank" collages, but it can completely change once it "settles" at a higher value, so it's risky to use a low setting as a 'draft mode'. The thing is, I actually prefer the sloppy version in some ways. So it can't hurt to get a render of it to have extra material. But at higher values it more accurately infers the emotional inflection and the overall flow of the performance.

**Guidance (CFG)**
This setting controls how much your prompt actually matters. The higher it goes, the more tinny (but precise) the sound will be. Conversely, if you turn it down far enough, you'll hear the seed and the denoising pattern by themselves, unaffected by any attempt to persuade it via prompt.

Any higher than about 12 and it starts developing blue splotches in the spectrogram and squealing unbearably. Don't do this. It might be hurting inside. Be nice.

You may find that turning it down lower than the 7-8 midpoint actually increases the perceived fidelity of the audio. This is very situational and seems to depend on just how much work you did to imitate instrumental performances. Vibrato and legato playing seem to allow you to set lower values here, as it naturally "realizes" you're trying to play e.g. funk or classical music based on how you phrased the notes in the sketch.

Past 4-5, you start to hear "MP3 artifacts" and unpleasant highs in the generation, but also notice more hyper-realism. Strings sound like strings more consistently.

2-3 tends to make things sound more adventurous and lively, but very disjointed texturally. You may hear a lovely trombone for 1 note, then it is a guitar! This is a place where a lot of magic can happen. A value this low is the model's "first take".. blurry and full of spurious extra limbs!

## Rendering
![instructions_shreddin3](https://user-images.githubusercontent.com/14255864/218354408-29d6e2c2-d091-451d-8eca-0811af1a2f19.png)

Saving the audio should be straightforward - Just make sure to set the duration of the render to the number of seconds of the song (helpfully displayed above the duration field), and click the big red Riff button. Then save it from the menu on the audio player that shows up at the very bottom, after everything is generated and concatenated.

## Editing

![instructions_shreddin4](https://user-images.githubusercontent.com/14255864/218354412-3a369050-fad9-40b4-b807-c37784635005.png)

To create most of the examples on this page, I rendered each prompt twice, with slightly different settings. Then I dragged both files into my DAW (Ableton Live, but you can use any multi-track wave editor!) and panned them left and right to create a stereo image. Then, I crossfaded between the original raw input file, and the stereo mix, to show their contrast.

I'm sure you will be able to think of ways to incorporate this into your own musical exploration, but please remember that this is all still active research, and there are still issues remaining to be solved in legal and ethical areas regarding the use of large models. So don't throw away your professional sample libraries just yet. But it's a good time to start learning.

-----------------------


### About the author:

Jake "virt" Kaufman's composing and sound design credits include Shovel Knight, Shantae, and Double Dragon Neon. He likes to build [musical circuits in Factorio](https://www.youtube.com/watch?v=p6H2LOGGcZQ), and [mod his Otamatone](https://www.youtube.com/watch?v=T3u8p98gBpg). His music is available for free under a Creative Commons license on his [Bandcamp page](https://virt.bandcamp.com).
