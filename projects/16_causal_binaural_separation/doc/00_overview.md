# 00 · Overview

## The problem
Speech separators like Demucs, Wave-U-Net, and non-causal Conv-TasNet or
TF-GridNet decide what a sample is from a context that reaches into the
future (hundreds of ms in total). A hearing aid or a hearable can't wait for
that. The processed sound is mixed acoustically with the direct sound
leaking past the ear tip, and once the device delay goes beyond roughly
5–10 ms, wearers hear comb filtering, echo, or their own voice doubled. The
usual hearing-aid budget is **≤ 5 ms in total**, and that includes converters
and compute.

A device like this has to be **strictly causal**: each output sample may
depend only on the past and on a few ms of buffering. A binaural device must
also keep the target's interaural cues, because the brain uses ITD and ILD to
place the talker and to unmask speech in noise. A separator that collapses
both ears onto the same signal has thrown that away.

## What the project measures
One controlled sweep over the **algorithmic latency** L = 1…32 ms, where L is
the synthesis window plus any lookahead (see `02_methodology.md`):

1. **Oracle masks** at each L, using both symmetric and asymmetric windows.
   These show how much is lost purely because a short window can't resolve
   frequency, whatever the network.
2. **Causal learned models** at each L: an asymmetric-window GRU with deep
   filtering (`AsymDF`), the same network with a symmetric short window, and
   a causal binaural Conv-TasNet.
3. **Non-causal references**: the same networks, allowed to see the whole
   utterance. "Hearing the future" is the gap between these and the causal
   models.
4. **Lookahead split**: for a fixed 8 ms budget, spending it on a longer
   window versus spending it on lookahead frames.
5. **Real-time cost**: measured per-chunk CPU time at batch 1 on one thread.

## Hypothesis
Below about 8 ms, the quality loss comes mainly from *frequency resolution*
(a short synthesis window), not from the missing future. If that's right, an
asymmetric window (long analysis, short synthesis) plus recurrent past state
should keep the 4 ms causal model within **1 dB SI-SDRi** of the 32 ms causal
model, and the common-gain or cue-loss variants should keep target ΔILD
< 1 dB and ΔITD < 20 µs (the JNDs).

## Falsification
Any one of these rejects the hypothesis:
- At 4 ms, the best causal model is more than 3 dB SI-SDRi below the 32 ms
  causal model.
- Target cue errors exceed the JNDs.
- At a fixed budget, lookahead beats window length, meaning the gap follows
  the future rather than the resolution.

## Why "distortion-free" is not the goal
No causal model at a few ms is perfect, and an oracle mask already isn't. The
deliverable is the *price list*: dB of SI-SDRi and µs of ITD per ms of
latency saved, and which part of that price a better network can buy back.
