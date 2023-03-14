from google.colab import drive
drive.mount('/content/gdrive')

!pip install --upgrade git+https://github.com/craftmine1000/jukebox-saveopt.git

import jukebox
import torch as t
import librosa
import os
from IPython.display import Audio
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample, \
                           load_prompts
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache

def init():
    global model

    device = 0 if torch.cuda.is_available() else -1


def inference(model_inputs: dict) -> dict:
    return {'wtf': 'Working up to here'}
    global model
    text = model_inputs.get('text', None)
    voice = model_inputs.get('voice', 'random')
    preset = model_inputs.get('preset', 'fast')
    model_type = model_inputs.get('type', None)
    n_samples = model_inputs.get('samples', None)
    save_path = model_inputs.get('path', None)

    try:
        if device is not None:
            pass
    except NameError:
        rank, local_rank, device = setup_dist_from_mpi()
    model = model_type if model_type is not None else "5b_lyrics"
    if model == '5b':
        your_lyrics = ""
    save_and_load_models_from_drive = False

    def load_5b_vqvae():
        if os.path.exists("/root/.cache/jukebox/models/5b/vqvae.pth.tar") == False:
            if os.path.exists("/content/gdrive/MyDrive/jukebox/models/5b/vqvae.pth.tar") == False:
                print("5b_vqvae not stored in Google Drive. Downloading for the first time.")
                # !wget https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar -O /content/gdrive/MyDrive/jukebox/models/5b/vqvae.pth.tar
            else:
                print("5b_vqvae stored in Google Drive.")
            print('Copying 5b VQVAE')
            # !pv /content/gdrive/MyDrive/jukebox/models/5b/vqvae.pth.tar > /root/.cache/jukebox/models/5b/vqvae.pth.tar

    def load_1b_lyrics_level2():
        if os.path.exists("/root/.cache/jukebox/models/1b_lyrics/prior_level_2.pth.tar") == False:
            if os.path.exists("/content/gdrive/MyDrive/jukebox/models/1b_lyrics/prior_level_2.pth.tar") == False:
                print(
                    "1b_lyrics_level_2 not stored in Google Drive. Downloading for the first time. This will take a few more minutes.")
                # !wget https://openaipublic.azureedge.net/jukebox/models/1b_lyrics/prior_level_2.pth.tar -O /content/gdrive/MyDrive/jukebox/models/1b_lyrics/prior_level_2.pth.tar
            else:
                print("1b_lyrics_level_2 stored in Google Drive.")
            print("Copying 1B_Lyrics Level 2")
            # !pv /content/gdrive/MyDrive/jukebox/models/1b_lyrics/prior_level_2.pth.tar > /root/.cache/jukebox/models/1b_lyrics/prior_level_2.pth.tar

    def load_5b_lyrics_level2():
        if os.path.exists("/root/.cache/jukebox/models/5b_lyrics/prior_level_2.pth.tar") == False:
            if os.path.exists("/content/gdrive/MyDrive/jukebox/models/5b_lyrics/prior_level_2.pth.tar") == False:
                print(
                    "5b_lyrics_level_2 not stored in Google Drive. Downloading for the first time. This will take up to 10-15 minutes.")
                # !wget https://openaipublic.azureedge.net/jukebox/models/5b_lyrics/prior_level_2.pth.tar -O /content/gdrive/MyDrive/jukebox/models/5b_lyrics/prior_level_2.pth.tar
            else:
                print("5b_lyrics_level_2 stored in Google Drive.")
            print("Copying 5B_Lyrics Level 2")
            # !pv /content/gdrive/MyDrive/jukebox/models/5b_lyrics/prior_level_2.pth.tar > /root/.cache/jukebox/models/5b_lyrics/prior_level_2.pth.tar

    def load_5b_level1():
        if os.path.exists('/root/.cache/jukebox/models/5b/prior_level_1.pth.tar') == False:
            if os.path.exists("/content/gdrive/MyDrive/jukebox/models/5b/prior_level_1.pth.tar") == False:
                print(
                    "5b_level_1 not stored in Google Drive. Downloading for the first time. This may take a few more minutes.")
                # !wget https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_1.pth.tar -O /content/gdrive/MyDrive/jukebox/models/5b/prior_level_1.pth.tar
            else:
                print("5b_level_1 stored in Google Drive.")
            print("Copying 5B Level 1")
            # !pv /content/gdrive/MyDrive/jukebox/models/5b/prior_level_1.pth.tar > /root/.cache/jukebox/models/5b/prior_level_1.pth.tar

    def load_5b_level0():
        if os.path.exists('/root/.cache/jukebox/models/5b/prior_level_0.pth.tar') == False:
            if os.path.exists("/content/gdrive/MyDrive/jukebox/models/5b/prior_level_0.pth.tar") == False:
                print(
                    "5b_level_0 not stored in Google Drive. Downloading for the first time. This may take a few minutes.")
                # !wget https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_0.pth.tar -O /content/gdrive/MyDrive/jukebox/models/5b/prior_level_0.pth.tar
            else:
                print("5b_level_0 stored in Google Drive.")
            print("Copying 5B Level 0")
            # !pv /content/gdrive/MyDrive/jukebox/models/5b/prior_level_0.pth.tar > /root/.cache/jukebox/models/5b/prior_level_0.pth.tar

    def load_5b_level2():
        if os.path.exists('/root/.cache/jukebox/models/5b/prior_level_2.pth.tar') == False:
            if os.path.exists("/content/gdrive/MyDrive/jukebox/models/5b/prior_level_2.pth.tar") == False:
                print(
                    "5b_level_2 not stored in Google Drive. Downloading for the first time. This will take up to 10-15 minutes.")
                # !wget https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar -O /content/gdrive/MyDrive/jukebox/models/5b/prior_level_2.pth.tar
            else:
                print("5b_level_2 stored in Google Drive.")
        print("Copying 5B Level 2")
        # !pv /content/gdrive/MyDrive/jukebox/models/5b/prior_level_2.pth.tar > /root/.cache/jukebox/models/5b/prior_level_2.pth.tar

    if save_and_load_models_from_drive == True:
        if model == '5b_lyrics':
            load_5b_vqvae()
            load_5b_lyrics_level2()
            load_5b_level1()
            load_5b_level0()
        if model == '5b':
            load_5b_vqvae()
            load_5b_level2()
            load_5b_level1()
            load_5b_level0()
        elif model == '1b_lyrics':
            load_5b_vqvae()
            load_1b_lyrics_level2()
            load_5b_level1()
            load_5b_level0()
    hps = Hyperparams()
    hps.sr = 44100
    hps.n_samples = n_samples if n_samples is not None else 2
    hps.name = save_path if save_path is not None else '/content/gdrive/MyDrive/Project_1'
    chunk_size = 64 if model in ('5b', '5b_lyrics') else 128
    gpu_info = nvidia - smi - L
    if gpu_info[0].find('Tesla T4') >= 0:
        max_batch_size = 2
        print('Tesla T4 detected, max_batch_size set to 2')
    elif gpu_info[0].find('Tesla K80') >= 0:
        max_batch_size = 8
        print('Tesla K80 detected, max_batch_size set to 8')
    elif gpu_info[0].find('Tesla P100') >= 0:
        max_batch_size = 3
        print('Tesla P100 detected, max_batch_size set to 3')
    elif gpu_info[0].find('Tesla V100') >= 0:
        max_batch_size = 3
        print('Tesla V100 detected, max_batch_size set to 3')
    elif gpu_info[0].find('A100') >= 0:
        max_batch_size = 6
        print('Tesla A100 detected, max_batch_size set to 6 (experimental)')
    else:
        max_batch_size = 3
        print('Different GPU detected, max_batch_size set to 3.')
    hps.levels = 3
    speed_upsampling = True
    if speed_upsampling == True:
        hps.hop_fraction = [1, 1, .125]
    else:
        hps.hop_fraction = [.5, .5, .125]

    vqvae, *priors = MODELS[model]
    vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=1048576)), device)
    top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)
    mode = 'primed'
    if mode == 'ancestral':
        codes_file = None
        audio_file = None
        prompt_length_in_seconds = None
    if mode == 'primed':
        codes_file = None
        # Specify an audio file here.
        audio_file = '/content/gdrive/MyDrive/your_file.wav'
        # Specify how many seconds of audio to prime on.
        prompt_length_in_seconds = 10

    sample_length_in_seconds = 70
    if sample_length_in_seconds < 24:
        sample_length_in_seconds = 24
        print('Chosen sample length too low. Automatically set to 24 seconds.')
    if os.path.exists(hps.name):
        # Identify the lowest level generated and continue from there.
        for level in [0, 1, 2]:
            data = f"{hps.name}/level_{level}/data.pth.tar"
            if os.path.isfile(data):
                codes_file = data
                if int(sample_length_in_seconds) > int(librosa.get_duration(filename=f'{hps.name}/level_2/item_0.wav')):
                    mode = 'continue'
                else:
                    mode = 'upsample'
                break

    print('mode is now ' + mode)
    if mode == 'continue':
        print('Continuing from level 2')
    if mode == 'upsample':
        print('Upsampling from level ' + str(level))

    sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file,
                                  prompt_length_in_seconds=prompt_length_in_seconds))

    if mode == 'upsample':
        sample_length_in_seconds = int(librosa.get_duration(filename=f'{hps.name}/level_{level}/item_0.wav'))
        data = t.load(sample_hps.codes_file, map_location='cpu')
        zs = [z.cpu() for z in data['zs']]
        hps.n_samples = zs[-1].shape[0]

    if mode == 'continue':
        data = t.load(sample_hps.codes_file, map_location='cpu')
        zs = [z.cpu() for z in data['zs']]
        hps.n_samples = zs[-1].shape[0]

    hps.sample_length = (int(sample_length_in_seconds * hps.sr) // top_prior.raw_to_tokens) * top_prior.raw_to_tokens
    assert hps.sample_length >= top_prior.n_ctx * top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

    # Note: Metas can contain different prompts per sample.
    # By default, all samples use the same prompt.

    select_artist = "the beatles"
    select_genre = "pop rock"
    metas = [dict(artist=select_artist,
                  genre=select_genre,
                  total_length=hps.sample_length,
                  offset=0,
                  lyrics=your_lyrics,
                  ),
             ] * hps.n_samples
    labels = [None, None, top_prior.labeller.get_batch_labels(metas, 'cuda')]

    sampling_temperature = .98

    if gpu_info[0].find('Tesla T4') >= 0:
        lower_batch_size = 14
        print('Tesla T4 detected, lower_batch_size set to 12')
    elif gpu_info[0].find('Tesla K80') >= 0:
        lower_batch_size = 8
        print('Tesla K80 detected, lower_batch_size set to 8')
    elif gpu_info[0].find('Tesla P100') >= 0:
        lower_batch_size = 16
        print('Tesla P100 detected, lower_batch_size set to 16')
    elif gpu_info[0].find('Tesla V100') >= 0:
        lower_batch_size = 16
        print('Tesla V100 detected, lower_batch_size set to 16')
    elif gpu_info[0].find('A100') >= 0:
        lower_batch_size = 48
        print('Tesla A100 detected, lower_batch_size set to 48 (experimental)')
    else:
        lower_batch_size = 8
        print('Different GPU detected, lower_batch_size set to 8.')
    lower_level_chunk_size = 32
    sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=lower_batch_size,
                            chunk_size=lower_level_chunk_size),
                       dict(temp=0.99, fp16=True, max_batch_size=lower_batch_size,
                            chunk_size=lower_level_chunk_size),
                       dict(temp=sampling_temperature, fp16=True,
                            max_batch_size=max_batch_size, chunk_size=chunk_size)]

    if sample_hps.mode == 'ancestral':
        zs = [t.zeros(hps.n_samples, 0, dtype=t.long, device='cpu') for _ in range(len(priors))]
        zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
    elif sample_hps.mode == 'upsample':
        assert sample_hps.codes_file is not None
        # Load codes.
        data = t.load(sample_hps.codes_file, map_location='cpu')
        zs = [z.cpu() for z in data['zs']]
        assert zs[-1].shape[0] == hps.n_samples, f"Expected bs = {hps.n_samples}, got {zs[-1].shape[0]}"
        del data
        print('One click upsampling!')
    elif sample_hps.mode == 'primed':
        assert sample_hps.audio_file is not None
        audio_files = sample_hps.audio_file.split(',')
        duration = (
                               int(sample_hps.prompt_length_in_seconds * hps.sr) // top_prior.raw_to_tokens) * top_prior.raw_to_tokens
        x = load_prompts(audio_files, duration, hps)
        zs = top_prior.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
        zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
    elif sample_hps.mode == 'continue':
        data = t.load(sample_hps.codes_file, map_location='cpu')
        zs = [z.cuda() for z in data['zs']]
        zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
    else:
        raise ValueError(f'Unknown sample mode {sample_hps.mode}.')

    # Set this False if you are on a local machine that has enough memory (this allows you to do the
    # lyrics alignment visualization during the upsampling stage). For a hosted runtime,
    # we'll need to go ahead and delete the top_prior if you are using the 5b_lyrics model.
    if True:
        del top_prior
        empty_cache()
        top_prior = None
    upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1]]
    labels[:2] = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers]

    zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)

    disconnect_runtime_after_finish = True
    if disconnect_runtime_after_finish == True:
        from google.colab import runtime
        runtime.unassign()
