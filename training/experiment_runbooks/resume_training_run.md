# Resume training after an instance is deleted

Supplemental reference for Verda. Use this when your instance gets deleted mid-run (Spot-preempted, out of budget, or deleted manually) and you still have the block volume. 

The block volume is untouched by instance deletion. Everything under `~` that lived on the block volume (the repo checkout, `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) survives. The ephemeral root disk — OS packages, pip installs, and `~/.bashrc` config — is gone.

Note that Verda block volumes are recoverable for 72 hours after deletion.

## 1. Create a new instance

On Verda.com, provision a fresh Verda instance with the same specs and the same image as the original (Ubuntu 26.04, CUDA 13.2, RTX 6000 Ada). 

Ensure that your instance is located in the location (eg. FIN-01, FIN-02) as the existing block volume. If it isn't, you can move the block volume to the same location as your instance, by selecting block volume settings in the Verda.com admin.

Once your instance and block volume are in the same location, in the Instance creation workflow, choose the **existing block volume** to attach. It mounts automatically under `~`.


## 2. Verify the data survived

```sh
ls ~
# expect: liver-us-detection-baselines  nnUNet_preprocessed  nnUNet_raw  nnUNet_results
```

## 3. Restore the shell environment

Add the `nnUNet_*` exports and history capture back to `~/.bashrc` (the old root disk is gone):

```sh
cat >> ~/.bashrc << 'ENVEOF'
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/liver-us-detection-baselines/training/custom_trainers"
ENVEOF

cat >> ~/.bashrc << 'PROMPTEOF'
PS1='\[\e[38;5;208m\]\u@\h:\w \t \[\e[0m\]\$ '
export HISTTIMEFORMAT='%F %T '
export PROMPT_COMMAND='history -a'
PROMPTEOF

source ~/.bashrc
```

## 4. Reinstall system and Python dependencies

```sh
apt update
apt install python3-pip unzip python-is-python3 tmux -y

rm -f /usr/lib/python3/dist-packages/typing_extensions.py
rm -rf /usr/lib/python3/dist-packages/typing_extensions-*.dist-info
rm -rf /usr/lib/python3/dist-packages/idna*
rm -rf /usr/lib/python3/dist-packages/click /usr/lib/python3/dist-packages/click-*.dist-info

cd ~/liver-us-detection-baselines
pip install -r requirements.txt --break-system-packages
pip install "nnunetv2==2.8.1" idna --break-system-packages
```

## 5. Verify the GPU

```sh
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 6. Resume training

```sh
tmux new -s training
```

Then, inside the tmux session:

```sh
cd ~/liver-us-detection-baselines
bash training/run_milestones_pilot.sh 2>&1 | tee run_milestones_pilot.log
```

Detach with `Ctrl+b` `d`; reattach with `tmux attach -t training`.

The runner uses `nnUNetv2_train --c`, so training resumes from the last checkpoint under `nnUNet_results` if it got far enough. Any unpreprocessed data or unfinished seeds are simply (re)done from scratch.
