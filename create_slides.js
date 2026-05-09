const pptxgen = require("pptxgenjs");
const path = require("path");

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Tajaddin Gafarov, Nicholas Kovacs, Vivekanandhan Kathirvel";
pres.title = "Sequence Modeling for Offline Reinforcement Learning";

// ── Color Palette: Midnight Executive ──
const C = {
  navy:    "1E2761",
  darkBg:  "141A3A",
  ice:     "CADCFC",
  white:   "FFFFFF",
  accent:  "4A90D9",
  accent2: "6CB4EE",
  gray:    "8E99A4",
  lightBg: "F4F6FA",
  green:   "27AE60",
  red:     "E74C3C",
  orange:  "F39C12",
  card:    "1B2550",
};

// ── Helper functions ──
const makeShadow = () => ({ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.15 });

function addFooter(slide, slideNum, total) {
  slide.addText("Sequence Modeling for Offline RL", {
    x: 0.5, y: 5.2, w: 5, h: 0.35, fontSize: 9, color: C.gray, fontFace: "Calibri"
  });
  slide.addText(`${slideNum} / ${total}`, {
    x: 8.5, y: 5.2, w: 1, h: 0.35, fontSize: 9, color: C.gray, fontFace: "Calibri", align: "right"
  });
}

function addSectionHeader(slide, title) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.3, w: 0.08, h: 0.55, fill: { color: C.accent }
  });
  slide.addText(title, {
    x: 0.75, y: 0.3, w: 8.5, h: 0.55, fontSize: 28, fontFace: "Georgia",
    color: C.navy, bold: true, margin: 0
  });
}

const TOTAL = 18;
let sn = 0;

// ════════════════════════════════════════════════════════════════
// SLIDE 1: Title
// ════════════════════════════════════════════════════════════════
sn++;
let s1 = pres.addSlide();
s1.background = { color: C.darkBg };
// Top accent bar
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
// Title
s1.addText("Sequence Modeling for\nOffline Reinforcement Learning", {
  x: 0.8, y: 1.0, w: 8.4, h: 2.2, fontSize: 40, fontFace: "Georgia",
  color: C.white, bold: true, lineSpacingMultiple: 1.15
});
// Subtitle line
s1.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 3.3, w: 2.5, h: 0.04, fill: { color: C.accent } });
s1.addText("Final Project Presentation", {
  x: 0.8, y: 3.5, w: 5, h: 0.5, fontSize: 18, fontFace: "Calibri", color: C.ice
});
// Authors
s1.addText([
  { text: "Tajaddin Gafarov", options: { bold: true, breakLine: true } },
  { text: "Nicholas Kovacs", options: { bold: true, breakLine: true } },
  { text: "Vivekanandhan Kathirvel", options: { bold: true } }
], { x: 0.8, y: 4.1, w: 5, h: 1.2, fontSize: 14, fontFace: "Calibri", color: C.accent2 });
// Course
s1.addText("Graduate Deep Learning  |  Spring 2026", {
  x: 5, y: 5.0, w: 4.5, h: 0.4, fontSize: 11, fontFace: "Calibri", color: C.gray, align: "right"
});
s1.addNotes(`Good [morning/afternoon] everyone. Today we're presenting our final project on Sequence Modeling for Offline Reinforcement Learning. Our team — Tajaddin, Nicholas, and Vivek — has been working on applying transformer architectures to the offline RL problem, specifically using the Decision Transformer framework. We'll walk you through our approach, results, and analysis.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 2: Project Goal
// ════════════════════════════════════════════════════════════════
sn++;
let s2 = pres.addSlide();
s2.background = { color: C.lightBg };
addSectionHeader(s2, "Project Goal");

// Problem card
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.2, w: 4.3, h: 3.5, fill: { color: C.white }, shadow: makeShadow()
});
s2.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.2, w: 4.3, h: 0.06, fill: { color: C.red } });
s2.addText("The Problem", {
  x: 0.8, y: 1.4, w: 3.7, h: 0.45, fontSize: 18, fontFace: "Georgia", color: C.navy, bold: true
});
s2.addText([
  { text: "Traditional RL is unstable", options: { bullet: true, breakLine: true, bold: true } },
  { text: "Requires thousands of live trial-and-error interactions", options: { bullet: true, breakLine: true } },
  { text: "Complex Bellman equations are hard to optimize", options: { bullet: true, breakLine: true } },
  { text: "Offline RL: learn from fixed logs with no new data collection", options: { bullet: true } }
], { x: 0.8, y: 2.0, w: 3.7, h: 2.5, fontSize: 13, fontFace: "Calibri", color: "444444", paraSpaceAfter: 6 });

// Solution card
s2.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 1.2, w: 4.3, h: 3.5, fill: { color: C.white }, shadow: makeShadow()
});
s2.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.2, w: 4.3, h: 0.06, fill: { color: C.green } });
s2.addText("Our Solution", {
  x: 5.5, y: 1.4, w: 3.7, h: 0.45, fontSize: 18, fontFace: "Georgia", color: C.navy, bold: true
});
s2.addText([
  { text: "Frame offline RL as sequence modeling", options: { bullet: true, breakLine: true, bold: true } },
  { text: "Use causal transformers instead of value-based methods", options: { bullet: true, breakLine: true } },
  { text: "Self-attention connects early actions to late rewards", options: { bullet: true, breakLine: true } },
  { text: "Condition on (Return-to-Go, State, Action) to predict optimal actions", options: { bullet: true } }
], { x: 5.5, y: 2.0, w: 3.7, h: 2.5, fontSize: 13, fontFace: "Calibri", color: "444444", paraSpaceAfter: 6 });

addFooter(s2, sn, TOTAL);
s2.addNotes(`The core problem we're tackling is that traditional reinforcement learning algorithms are unstable and require thousands of live trial-and-error attempts. Our solution is to reframe offline RL as a sequence modeling problem. Instead of using complex Bellman equations, we use causal transformers. The key insight is that self-attention naturally connects early actions to rewards received much later, solving what's called the long-term credit assignment problem. Our models take in a sequence of return-to-go, state, and action tokens to predict the next optimal action. This is fundamentally a Markov Decision Process — the agent observes a state, takes an action, receives a reward, and transitions. But in offline RL, we must learn entirely from a fixed dataset of past experience.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 3: Input and Ground Truth
// ════════════════════════════════════════════════════════════════
sn++;
let s3 = pres.addSlide();
s3.background = { color: C.lightBg };
addSectionHeader(s3, "Input and Ground Truth");

// Left: inputs
s3.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.2, w: 4.3, h: 3.8, fill: { color: C.white }, shadow: makeShadow()
});
s3.addText("Model Inputs", {
  x: 0.8, y: 1.35, w: 3.7, h: 0.4, fontSize: 18, fontFace: "Georgia", color: C.navy, bold: true
});
s3.addText([
  { text: "State: ", options: { bold: true } },
  { text: "11-dim vector of robot joint angles, velocities", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Action: ", options: { bold: true } },
  { text: "3-dim continuous motor torque commands", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Return-to-Go: ", options: { bold: true } },
  { text: "Cumulative future reward from current timestep", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Timestep: ", options: { bold: true } },
  { text: "Position encoding for temporal ordering", options: {} }
], { x: 0.8, y: 1.9, w: 3.7, h: 2.8, fontSize: 13, fontFace: "Calibri", color: "444444" });

// Right: ground truth + sequence
s3.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 1.2, w: 4.3, h: 1.7, fill: { color: C.white }, shadow: makeShadow()
});
s3.addText("Ground Truth", {
  x: 5.5, y: 1.35, w: 3.7, h: 0.35, fontSize: 18, fontFace: "Georgia", color: C.navy, bold: true
});
s3.addText("The actual action taken by the agent in the offline dataset at each timestep.", {
  x: 5.5, y: 1.8, w: 3.7, h: 0.9, fontSize: 13, fontFace: "Calibri", color: "444444"
});

// Sequence flow
s3.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 3.15, w: 4.3, h: 1.85, fill: { color: C.navy }, shadow: makeShadow()
});
s3.addText("Token Sequence", {
  x: 5.5, y: 3.3, w: 3.7, h: 0.35, fontSize: 16, fontFace: "Georgia", color: C.white, bold: true
});
// Token boxes
const tokenY = 3.85;
const tokens = [
  { label: "RTG", color: C.green, x: 5.5 },
  { label: "State", color: C.accent, x: 6.7 },
  { label: "Action", color: C.orange, x: 7.9 },
];
tokens.forEach(t => {
  s3.addShape(pres.shapes.RECTANGLE, {
    x: t.x, y: tokenY, w: 1.0, h: 0.45, fill: { color: t.color },
    rectRadius: 0.05
  });
  s3.addText(t.label, {
    x: t.x, y: tokenY, w: 1.0, h: 0.45, fontSize: 12, fontFace: "Calibri",
    color: C.white, bold: true, align: "center", valign: "middle"
  });
});
s3.addText("Repeating triplet at each timestep t", {
  x: 5.5, y: 4.45, w: 3.7, h: 0.4, fontSize: 11, fontFace: "Calibri", color: C.ice, italic: true
});

addFooter(s3, sn, TOTAL);
s3.addNotes(`For our model inputs, the state is an 11-dimensional vector representing the Hopper robot's joint angles and velocities. The action is a 3-dimensional continuous motor torque command. The Return-to-Go is the cumulative future reward from the current timestep — this is what tells the Decision Transformer how well we want to perform. The ground truth is simply the actual action taken by the agent in the offline dataset. The key design is this repeating token triplet: Return-to-Go, then State, then Action — fed sequentially into the transformer at each timestep.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 4: Dataset Description
// ════════════════════════════════════════════════════════════════
sn++;
let s4 = pres.addSlide();
s4.background = { color: C.lightBg };
addSectionHeader(s4, "Dataset Description");

// Stats cards
const stats = [
  { label: "Trajectories", value: "2,041", color: C.accent },
  { label: "Transitions", value: "402K", color: C.green },
  { label: "State Dim", value: "11", color: C.orange },
  { label: "Action Dim", value: "3", color: C.red },
];
stats.forEach((st, i) => {
  const cx = 0.5 + i * 2.35;
  s4.addShape(pres.shapes.RECTANGLE, {
    x: cx, y: 1.2, w: 2.1, h: 1.3, fill: { color: C.white }, shadow: makeShadow()
  });
  s4.addText(st.value, {
    x: cx, y: 1.3, w: 2.1, h: 0.7, fontSize: 32, fontFace: "Georgia",
    color: st.color, bold: true, align: "center", valign: "middle"
  });
  s4.addText(st.label, {
    x: cx, y: 2.05, w: 2.1, h: 0.35, fontSize: 12, fontFace: "Calibri",
    color: C.gray, align: "center"
  });
});

// Description card
s4.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 2.8, w: 9, h: 2.2, fill: { color: C.white }, shadow: makeShadow()
});
s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.8, w: 0.08, h: 2.2, fill: { color: C.accent } });
s4.addText("D4RL: hopper-medium-replay-v2", {
  x: 0.85, y: 2.9, w: 8.3, h: 0.4, fontSize: 18, fontFace: "Georgia", color: C.navy, bold: true
});
s4.addText([
  { text: "Public benchmark dataset from the D4RL suite (Fu et al., 2020)", options: { bullet: true, breakLine: true } },
  { text: "Medium-Replay: Contains the full replay buffer of a partially trained policy", options: { bullet: true, breakLine: true } },
  { text: "Chaotic mix of successful runs AND complete failures — forces the model to learn what works", options: { bullet: true, breakLine: true } },
  { text: "Return range: -1.4 to 3,192.9 (avg: 467.3 +/- 511.0) — high variance dataset", options: { bullet: true, breakLine: true } },
  { text: "Environment: MuJoCo Hopper — a one-legged robot learning to hop forward", options: { bullet: true } }
], { x: 0.85, y: 3.4, w: 8.3, h: 1.5, fontSize: 13, fontFace: "Calibri", color: "444444", paraSpaceAfter: 4 });

addFooter(s4, sn, TOTAL);
s4.addNotes(`We use the D4RL hopper-medium-replay-v2 dataset. This is a public benchmark dataset, specifically the replay buffer variant. What makes this dataset interesting and challenging is that it contains a chaotic mix of both successful runs and complete failures from a partially trained policy. The dataset has 2,041 trajectories totaling about 402,000 transitions. The return ranges from negative 1.4 all the way up to 3,192, with a very high standard deviation of 511. This forces our models to actually learn which behaviors work and which don't, rather than just copying an expert. The environment is MuJoCo Hopper — a simulated one-legged robot that needs to learn to hop forward as fast as possible.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 5: Data Processing
// ════════════════════════════════════════════════════════════════
sn++;
let s5 = pres.addSlide();
s5.background = { color: C.lightBg };
addSectionHeader(s5, "Data Processing");

// Step cards in 2x2 grid
const steps = [
  { num: "1", title: "HDF5 Loading", desc: "Raw dataset loaded from HDF5 file containing observations, actions, rewards, terminals, and timeouts. No external d4rl dependency needed.", color: C.accent },
  { num: "2", title: "Trajectory Segmentation", desc: "Flat arrays split into individual trajectories at terminal/timeout boundaries. Returns-to-go computed via reverse cumulative sum of rewards.", color: C.green },
  { num: "3", title: "Context Windowing", desc: "Sliding window of length K=20 timesteps. Each sample is a (state, action, RTG, timestep) tuple. Shorter sequences are zero-padded on the left.", color: C.orange },
  { num: "4", title: "Train/Val Split", desc: "90/10 random split. No data leakage: splits are at the sample level within pre-segmented trajectories. Test evaluation done in live MuJoCo environment.", color: C.red },
];
steps.forEach((st, i) => {
  const col = i % 2;
  const row = Math.floor(i / 2);
  const cx = 0.5 + col * 4.7;
  const cy = 1.2 + row * 2.05;
  s5.addShape(pres.shapes.RECTANGLE, {
    x: cx, y: cy, w: 4.4, h: 1.8, fill: { color: C.white }, shadow: makeShadow()
  });
  // Number circle
  s5.addShape(pres.shapes.OVAL, {
    x: cx + 0.15, y: cy + 0.15, w: 0.5, h: 0.5, fill: { color: st.color }
  });
  s5.addText(st.num, {
    x: cx + 0.15, y: cy + 0.15, w: 0.5, h: 0.5, fontSize: 18, fontFace: "Georgia",
    color: C.white, bold: true, align: "center", valign: "middle"
  });
  s5.addText(st.title, {
    x: cx + 0.8, y: cy + 0.15, w: 3.3, h: 0.4, fontSize: 16, fontFace: "Georgia",
    color: C.navy, bold: true, margin: 0
  });
  s5.addText(st.desc, {
    x: cx + 0.15, y: cy + 0.7, w: 4.1, h: 1.0, fontSize: 12, fontFace: "Calibri", color: "555555"
  });
});

addFooter(s5, sn, TOTAL);
s5.addNotes(`For data processing, we follow four steps. First, we load the raw dataset from an HDF5 file — we built our own loader to avoid Windows compatibility issues with the d4rl package. Second, we segment the flat arrays into individual trajectories by splitting at terminal or timeout boundaries, and compute the returns-to-go using a reverse cumulative sum of rewards. Third, we apply context windowing — each training sample is a sliding window of K equals 20 timesteps containing state, action, return-to-go, and timestep tokens. Sequences shorter than K are zero-padded on the left. Fourth, we do a 90/10 train-validation split. There's no data leakage because the split is at the sample level, and our test evaluation is done entirely in the live MuJoCo environment, not on held-out offline data.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 6: Methodology — Baseline (BC)
// ════════════════════════════════════════════════════════════════
sn++;
let s6 = pres.addSlide();
s6.background = { color: C.lightBg };
addSectionHeader(s6, "Baseline: Behavior Cloning (BC)");

// Architecture card
s6.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.2, w: 5.5, h: 3.8, fill: { color: C.white }, shadow: makeShadow()
});
s6.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.2, w: 0.08, h: 3.8, fill: { color: C.gray } });
s6.addText("Architecture", {
  x: 0.85, y: 1.3, w: 4.8, h: 0.4, fontSize: 18, fontFace: "Georgia", color: C.navy, bold: true
});
s6.addText([
  { text: "Causal GPT-style transformer", options: { bullet: true, breakLine: true } },
  { text: "Input: past states only (ignores rewards)", options: { bullet: true, breakLine: true } },
  { text: "Output: predicted action at current timestep", options: { bullet: true, breakLine: true } },
  { text: "Loss: MSE between predicted and actual actions", options: { bullet: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Why it's a valid baseline:", options: { bold: true, breakLine: true } },
  { text: "Mimics average behavior from dataset", options: { bullet: true, indentLevel: 1, breakLine: true } },
  { text: "Beats random agent but cannot distinguish good vs. bad trajectories", options: { bullet: true, indentLevel: 1 } }
], { x: 0.85, y: 1.8, w: 4.8, h: 3.0, fontSize: 13, fontFace: "Calibri", color: "444444", paraSpaceAfter: 4 });

// Specs card
s6.addShape(pres.shapes.RECTANGLE, {
  x: 6.3, y: 1.2, w: 3.2, h: 3.8, fill: { color: C.navy }, shadow: makeShadow()
});
s6.addText("Model Specs", {
  x: 6.5, y: 1.35, w: 2.8, h: 0.4, fontSize: 16, fontFace: "Georgia", color: C.white, bold: true
});
const specs = [
  ["Parameters", "599,555"],
  ["Layers", "3"],
  ["Heads", "4"],
  ["Hidden Dim", "128"],
  ["Context (K)", "20"],
  ["Dropout", "0.1"],
  ["Optimizer", "AdamW"],
  ["LR", "1e-4"],
];
specs.forEach((sp, i) => {
  const sy = 1.9 + i * 0.37;
  s6.addText(sp[0], { x: 6.5, y: sy, w: 1.6, h: 0.3, fontSize: 11, fontFace: "Calibri", color: C.ice });
  s6.addText(sp[1], { x: 8.1, y: sy, w: 1.2, h: 0.3, fontSize: 11, fontFace: "Calibri", color: C.accent2, bold: true, align: "right" });
});

addFooter(s6, sn, TOTAL);
s6.addNotes(`Our baseline model is Behavior Cloning using a causal GPT-style transformer. The key difference from the Decision Transformer is that BC only sees past states — it completely ignores rewards and return-to-go. It simply predicts what action to take based on the state history, trained with mean squared error loss. This makes it a pure imitation learner — it mimics the average behavior in the dataset. It's a valid baseline because it beats a random agent, but it fundamentally cannot distinguish between good and bad trajectories in our mixed-quality dataset. The model has about 600,000 parameters with 3 transformer layers, 4 attention heads, and a hidden dimension of 128.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 7: Decision Transformer
// ════════════════════════════════════════════════════════════════
sn++;
let s7 = pres.addSlide();
s7.background = { color: C.lightBg };
addSectionHeader(s7, "Proposed Model 1: Decision Transformer");

s7.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.2, w: 5.5, h: 3.8, fill: { color: C.white }, shadow: makeShadow()
});
s7.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.2, w: 0.08, h: 3.8, fill: { color: C.accent } });
s7.addText("Key Innovations", {
  x: 0.85, y: 1.3, w: 4.8, h: 0.4, fontSize: 18, fontFace: "Georgia", color: C.navy, bold: true
});
s7.addText([
  { text: "Conditioned on target Return-to-Go", options: { bullet: true, breakLine: true, bold: true } },
  { text: "Can actively pursue high scores, not just imitate", options: { bullet: true, indentLevel: 1, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "Supervised sequence prediction (no Bellman backups)", options: { bullet: true, breakLine: true, bold: true } },
  { text: "Standard MSE loss, no value bootstrapping", options: { bullet: true, indentLevel: 1, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "Deterministic action output", options: { bullet: true, breakLine: true, bold: true } },
  { text: "Outputs one exact action per timestep", options: { bullet: true, indentLevel: 1, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "Credit assignment via attention", options: { bullet: true, breakLine: true, bold: true } },
  { text: "Self-attention connects actions to delayed rewards", options: { bullet: true, indentLevel: 1 } },
], { x: 0.85, y: 1.8, w: 4.8, h: 3.0, fontSize: 13, fontFace: "Calibri", color: "444444", paraSpaceAfter: 2 });

// Specs
s7.addShape(pres.shapes.RECTANGLE, {
  x: 6.3, y: 1.2, w: 3.2, h: 3.8, fill: { color: C.navy }, shadow: makeShadow()
});
s7.addText("Model Specs", {
  x: 6.5, y: 1.35, w: 2.8, h: 0.4, fontSize: 16, fontFace: "Georgia", color: C.white, bold: true
});
const dtSpecs = [
  ["Parameters", "733,443"],
  ["Layers", "3"],
  ["Heads", "4"],
  ["Hidden Dim", "128"],
  ["Context (K)", "20"],
  ["Loss", "MSE"],
  ["Scheduler", "CosineAnnealing"],
  ["Grad Clip", "0.25"],
];
dtSpecs.forEach((sp, i) => {
  const sy = 1.9 + i * 0.37;
  s7.addText(sp[0], { x: 6.5, y: sy, w: 1.6, h: 0.3, fontSize: 11, fontFace: "Calibri", color: C.ice });
  s7.addText(sp[1], { x: 8.1, y: sy, w: 1.2, h: 0.3, fontSize: 11, fontFace: "Calibri", color: C.accent2, bold: true, align: "right" });
});

addFooter(s7, sn, TOTAL);
s7.addNotes(`Our first proposed model is the Decision Transformer, based on Chen et al. 2021. The key innovation over the baseline is that it's conditioned on a target Return-to-Go. At test time, we can set a high target return and the model will try to produce actions that achieve that score. Unlike traditional RL, there are no Bellman backups or value bootstrapping — it's purely supervised sequence prediction using MSE loss. The model outputs one deterministic action per timestep. It has about 733,000 parameters — slightly more than BC because it has additional embedding layers for the return-to-go and action tokens. The credit assignment happens naturally through the self-attention mechanism connecting actions to delayed rewards.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 8: Online Decision Transformer
// ════════════════════════════════════════════════════════════════
sn++;
let s8 = pres.addSlide();
s8.background = { color: C.lightBg };
addSectionHeader(s8, "Proposed Model 2: Online Decision Transformer");

s8.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.2, w: 5.5, h: 3.8, fill: { color: C.white }, shadow: makeShadow()
});
s8.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.2, w: 0.08, h: 3.8, fill: { color: C.green } });
s8.addText("Upgrades over DT", {
  x: 0.85, y: 1.3, w: 4.8, h: 0.4, fontSize: 18, fontFace: "Georgia", color: C.navy, bold: true
});
s8.addText([
  { text: "Stochastic policy (Gaussian output)", options: { bullet: true, breakLine: true, bold: true } },
  { text: "Outputs mean + std instead of a single action", options: { bullet: true, indentLevel: 1, breakLine: true } },
  { text: "Enables exploration through action sampling", options: { bullet: true, indentLevel: 1, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "Sequence-level entropy bonus", options: { bullet: true, breakLine: true, bold: true } },
  { text: "Loss = NLL - 0.01 * Entropy", options: { bullet: true, indentLevel: 1, breakLine: true } },
  { text: "Encourages diverse strategy learning during offline phase", options: { bullet: true, indentLevel: 1, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "Online fine-tuning phase", options: { bullet: true, breakLine: true, bold: true } },
  { text: "Collects live rollouts in MuJoCo environment", options: { bullet: true, indentLevel: 1, breakLine: true } },
  { text: "Hindsight return relabeling on collected data", options: { bullet: true, indentLevel: 1 } },
], { x: 0.85, y: 1.8, w: 4.8, h: 3.0, fontSize: 13, fontFace: "Calibri", color: "444444", paraSpaceAfter: 2 });

// Training phases
s8.addShape(pres.shapes.RECTANGLE, {
  x: 6.3, y: 1.2, w: 3.2, h: 1.7, fill: { color: C.navy }, shadow: makeShadow()
});
s8.addText("Phase 1: Offline", {
  x: 6.5, y: 1.3, w: 2.8, h: 0.35, fontSize: 14, fontFace: "Georgia", color: C.accent2, bold: true
});
s8.addText("50 epochs on D4RL dataset\nLearns diverse strategies\nfrom mixed-quality data", {
  x: 6.5, y: 1.7, w: 2.8, h: 1.0, fontSize: 12, fontFace: "Calibri", color: C.ice
});

s8.addShape(pres.shapes.RECTANGLE, {
  x: 6.3, y: 3.15, w: 3.2, h: 1.85, fill: { color: "1B3A1B" }, shadow: makeShadow()
});
s8.addText("Phase 2: Online", {
  x: 6.5, y: 3.25, w: 2.8, h: 0.35, fontSize: 14, fontFace: "Georgia", color: C.green, bold: true
});
s8.addText("10 epochs of live rollouts\n10 rollouts per epoch\nHindsight relabeling turns\nexperience into training data", {
  x: 6.5, y: 3.65, w: 2.8, h: 1.2, fontSize: 12, fontFace: "Calibri", color: C.ice
});

addFooter(s8, sn, TOTAL);
s8.addNotes(`The Online Decision Transformer, based on Zheng et al. 2022, adds three key upgrades. First, it uses a stochastic policy — instead of outputting one exact action, it outputs a mean and standard deviation, then samples from that Gaussian distribution. This enables exploration. Second, it adds an entropy bonus to the loss function, which encourages the model to learn diverse strategies during offline training. Third, and most importantly, it has an online fine-tuning phase. After the 50 offline epochs, the model collects live rollouts in the MuJoCo environment. After each rollout, it uses hindsight return relabeling — retroactively assigning optimal return targets to the trajectory it just experienced. This turns real interactions into high-quality training data on the fly.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 9: Training Details & Reproducibility
// ════════════════════════════════════════════════════════════════
sn++;
let s9 = pres.addSlide();
s9.background = { color: C.lightBg };
addSectionHeader(s9, "Training Details & Reproducibility");

// Table
let tableHeader = [
  { text: "Setting", options: { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 12, fontFace: "Calibri" } },
  { text: "BC", options: { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 12, fontFace: "Calibri", align: "center" } },
  { text: "DT", options: { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 12, fontFace: "Calibri", align: "center" } },
  { text: "ODT", options: { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 12, fontFace: "Calibri", align: "center" } },
];
let tableRows = [
  ["Framework", "PyTorch 2.x", "PyTorch 2.x", "PyTorch 2.x"],
  ["Hardware", "RTX 4060 (CUDA)", "RTX 4060 (CUDA)", "RTX 4060 (CUDA)"],
  ["Epochs", "50", "50", "50 + 10 online"],
  ["Batch Size", "64", "64", "64"],
  ["Learning Rate", "1e-4", "1e-4", "1e-4 (1e-5 online)"],
  ["Optimizer", "AdamW", "AdamW", "AdamW"],
  ["Weight Decay", "1e-4", "1e-4", "1e-4"],
  ["Scheduler", "CosineAnnealing", "CosineAnnealing", "CosineAnnealing"],
  ["Gradient Clipping", "0.25", "0.25", "0.25"],
  ["Seed", "42", "42", "42"],
].map(row => row.map((cell, ci) => ({
  text: cell,
  options: {
    fontSize: 11, fontFace: "Calibri",
    fill: { color: C.white },
    color: ci === 0 ? C.navy : "444444",
    bold: ci === 0,
    align: ci === 0 ? "left" : "center",
  }
})));

s9.addTable([tableHeader, ...tableRows], {
  x: 0.5, y: 1.15, w: 9, colW: [2.5, 2.17, 2.17, 2.17],
  border: { pt: 0.5, color: "DDDDDD" },
  rowH: 0.35,
});

addFooter(s9, sn, TOTAL);
s9.addNotes(`For reproducibility, here are the exact training details. All three models were trained using PyTorch 2.x on an NVIDIA RTX 4060 GPU with CUDA. We used a batch size of 64, learning rate of 1e-4 with AdamW optimizer and cosine annealing scheduler. All models trained for 50 epochs, with ODT getting an additional 10 online fine-tuning epochs at a reduced learning rate of 1e-5. We used gradient clipping at 0.25 and a fixed random seed of 42 for reproducibility. The entire training pipeline runs on GPU — all tensor operations, model forward and backward passes are on CUDA.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 10: Training Loss Curves
// ════════════════════════════════════════════════════════════════
sn++;
let s10 = pres.addSlide();
s10.background = { color: C.lightBg };
addSectionHeader(s10, "Training Loss Curves");

s10.addImage({
  path: path.resolve("G:/Project 2/Main/outputs/all_loss_curves.png"),
  x: 0.3, y: 1.1, w: 9.4, h: 4.2,
  sizing: { type: "contain", w: 9.4, h: 4.2 }
});

addFooter(s10, sn, TOTAL);
s10.addNotes(`Here are the training loss curves for all three models. BC and DT both show clean MSE loss convergence — the validation loss decreases smoothly and plateaus without overfitting, indicating good generalization. ODT's loss looks different because it uses a negative log-likelihood plus entropy loss, so the values go negative as the model becomes more confident — this is expected behavior. All three models converge well by epoch 50. The key observation is that lower validation loss doesn't always correlate with higher D4RL evaluation scores — the environment evaluation is what truly matters for RL performance.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 11: D4RL Score Progression
// ════════════════════════════════════════════════════════════════
sn++;
let s11 = pres.addSlide();
s11.background = { color: C.lightBg };
addSectionHeader(s11, "D4RL Score Progression During Training");

s11.addImage({
  path: path.resolve("G:/Project 2/Main/outputs/d4rl_progression.png"),
  x: 0.3, y: 1.1, w: 9.4, h: 4.2,
  sizing: { type: "contain", w: 9.4, h: 4.2 }
});

addFooter(s11, sn, TOTAL);
s11.addNotes(`This plot shows the D4RL normalized score during training, evaluated every 5 epochs. The BC baseline in gray climbs steadily and reaches about 48 by epoch 45. The DT in blue is more volatile — this is because small changes in the transformer weights can significantly affect the generated trajectory quality, and the return conditioning makes it sensitive to the target return setting. The ODT offline phase in green plateaus around 14 to 17 — this is because the entropy bonus deliberately keeps the policy stochastic during offline training. But then the online phase in red shows a clear jump, with scores reaching up to 40 as the model fine-tunes on live experience.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 12: Performance Improvement Techniques
// ════════════════════════════════════════════════════════════════
sn++;
let s12 = pres.addSlide();
s12.background = { color: C.lightBg };
addSectionHeader(s12, "Performance Improvement Techniques");

// 3 technique cards
const techniques = [
  {
    num: "1", title: "State & RTG Normalization",
    effect: "Best: 16.2", baseline: "Baseline: 46.7",
    desc: "Normalized states to zero-mean, unit-variance and scaled returns-to-go. Lowered val loss (0.082 vs 0.085) but hurt evaluation score.",
    why: "Normalization helped the loss landscape but changed the RTG scale the model conditions on, reducing its ability to target high returns during evaluation.",
    color: C.accent
  },
  {
    num: "2", title: "LR Warm-up + Cosine Decay",
    effect: "Best: 39.4", baseline: "Baseline: 46.7",
    desc: "Linear warm-up for first 10% of training steps, then cosine decay to near-zero. Showed more stable mid-training performance.",
    why: "Warm-up prevented early gradient spikes but the slower ramp-up meant less training at full learning rate, slightly reducing peak performance.",
    color: C.green
  },
  {
    num: "3", title: "Context Length K=30",
    effect: "Best: 29.7", baseline: "Baseline: 46.7",
    desc: "Increased context window from K=20 to K=30, providing 50% more history. Achieved lowest val loss (0.076) of all experiments.",
    why: "Longer context needs more data/epochs to fully utilize. The additional capacity lowered loss but the model hadn't converged in evaluation by epoch 50.",
    color: C.orange
  },
];

techniques.forEach((t, i) => {
  const cy = 1.15 + i * 1.42;
  s12.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: cy, w: 9, h: 1.3, fill: { color: C.white }, shadow: makeShadow()
  });
  s12.addShape(pres.shapes.OVAL, {
    x: 0.7, y: cy + 0.1, w: 0.45, h: 0.45, fill: { color: t.color }
  });
  s12.addText(t.num, {
    x: 0.7, y: cy + 0.1, w: 0.45, h: 0.45, fontSize: 16, fontFace: "Georgia",
    color: C.white, bold: true, align: "center", valign: "middle"
  });
  s12.addText(t.title, {
    x: 1.3, y: cy + 0.08, w: 4, h: 0.35, fontSize: 15, fontFace: "Georgia",
    color: C.navy, bold: true, margin: 0
  });
  s12.addText(t.effect, {
    x: 7.8, y: cy + 0.08, w: 1.5, h: 0.2, fontSize: 11, fontFace: "Calibri",
    color: C.red, bold: true, align: "right", margin: 0
  });
  s12.addText(t.baseline, {
    x: 7.8, y: cy + 0.3, w: 1.5, h: 0.2, fontSize: 10, fontFace: "Calibri",
    color: C.gray, align: "right", margin: 0
  });
  s12.addText(t.desc, {
    x: 1.3, y: cy + 0.5, w: 4.5, h: 0.7, fontSize: 11, fontFace: "Calibri", color: "555555"
  });
  s12.addText(t.why, {
    x: 6.0, y: cy + 0.5, w: 3.3, h: 0.7, fontSize: 11, fontFace: "Calibri", color: C.navy, italic: true
  });
});

addFooter(s12, sn, TOTAL);
s12.addNotes(`We implemented three performance improvement techniques on the Decision Transformer. Important note: while all three techniques actually lowered the validation loss compared to baseline, none of them improved the D4RL evaluation score. This is a key finding.

Technique 1, State and RTG Normalization: normalizing helped the optimization landscape, achieving a lower validation loss. But it changed the scale of the return-to-go conditioning, so during evaluation, the model couldn't properly target high returns.

Technique 2, Learning Rate Warm-up: this showed more stable mid-training behavior and achieved 39.4 at its best evaluation. The warm-up prevented early gradient explosions but the slower ramp-up meant less effective training time.

Technique 3, Increased Context Length to K=30: achieved the lowest validation loss of ALL experiments at 0.076, showing the model learns better representations with more history. However, the longer context needs more epochs to fully converge in the actual environment.

The takeaway is that offline loss and online evaluation measure different things in RL. A model can fit the dataset better but still perform worse when actually controlling the robot.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 13: Main Results
// ════════════════════════════════════════════════════════════════
sn++;
let s13 = pres.addSlide();
s13.background = { color: C.lightBg };
addSectionHeader(s13, "Performance Evaluation");

// Results table
let resHeader = [
  { text: "Model", options: { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 13, fontFace: "Calibri" } },
  { text: "Mean Return", options: { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 13, fontFace: "Calibri", align: "center" } },
  { text: "Std Dev", options: { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 13, fontFace: "Calibri", align: "center" } },
  { text: "D4RL Score", options: { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 13, fontFace: "Calibri", align: "center" } },
];
let resRows = [
  ["BC (Baseline)", "1,478.8", "220.4", "46.1"],
  ["Decision Transformer", "795.1", "551.9", "25.1"],
  ["DT (RTG=3600)", "1,409.3", "—", "43.9"],
  ["ODT (Offline Only)", "536.3", "8.3", "17.1"],
  ["ODT (Online FT)", "1,691.8", "545.7", "52.6"],
  ["ODT (RTG=3000)", "2,055.6", "—", "63.8"],
].map((row, ri) => row.map((cell, ci) => ({
  text: cell,
  options: {
    fontSize: 12, fontFace: "Calibri",
    fill: { color: (ri === 5) ? "E8F5E9" : C.white },
    color: ci === 0 ? C.navy : (ri === 5 && ci === 3) ? C.green : "444444",
    bold: ci === 0 || (ri === 5),
    align: ci === 0 ? "left" : "center",
  }
})));

s13.addTable([resHeader, ...resRows], {
  x: 0.5, y: 1.15, w: 9, colW: [3.0, 2.0, 2.0, 2.0],
  border: { pt: 0.5, color: "DDDDDD" }, rowH: 0.42,
});

// Key finding callout
s13.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 4.0, w: 9, h: 0.9, fill: { color: C.navy }, shadow: makeShadow()
});
s13.addText([
  { text: "Key Finding:  ", options: { bold: true, color: C.accent2 } },
  { text: "ODT with online fine-tuning achieves 63.8 D4RL score (RTG=3000), outperforming all other approaches. BC < ODT confirmed.", options: { color: C.white } },
], { x: 0.8, y: 4.05, w: 8.4, h: 0.8, fontSize: 14, fontFace: "Calibri", valign: "middle" });

addFooter(s13, sn, TOTAL);
s13.addNotes(`Here are our main test results evaluated over 20 episodes. The BC baseline achieves a solid 46.1 D4RL score. The Decision Transformer at its default target return gets 25.1, but when we sweep the target return to 3600, it reaches 43.9, close to BC. The big story is ODT. During the offline phase alone, it only gets 17.1 because the entropy bonus keeps the policy deliberately stochastic. But after online fine-tuning, it jumps to 52.6, and with optimal return targeting at 3000, it reaches 63.8 — our best result. This confirms our hypothesis that BC is less than ODT. The return-to-go sweep is important: it shows the model has learned to condition on desired performance, which is the whole point of the Decision Transformer framework.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 14: Model Comparison Bar Chart
// ════════════════════════════════════════════════════════════════
sn++;
let s14 = pres.addSlide();
s14.background = { color: C.lightBg };
addSectionHeader(s14, "Comparison of Approaches");

s14.addImage({
  path: path.resolve("G:/Project 2/Main/outputs/model_comparison_bar.png"),
  x: 0.3, y: 1.1, w: 9.4, h: 4.2,
  sizing: { type: "contain", w: 9.4, h: 4.2 }
});

addFooter(s14, sn, TOTAL);
s14.addNotes(`This bar chart visualizes the comparison across all our approaches. The key takeaway is the progression: BC gives a solid baseline around 46 to 48. DT matches it when properly tuned but is more volatile. ODT with online fine-tuning clearly wins at 57.4 in the standard test and up to 63.8 with return sweep. The improvement techniques on DT show an interesting pattern — they all improved training loss but not necessarily evaluation performance, highlighting the gap between supervised fitting and actual RL control quality.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 15: Foundation Model Comparison
// ════════════════════════════════════════════════════════════════
sn++;
let s15 = pres.addSlide();
s15.background = { color: C.lightBg };
addSectionHeader(s15, "Foundation Model Comparison");

// Results
let fmHeader = [
  { text: "Model", options: { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 13, fontFace: "Calibri" } },
  { text: "Mean Return", options: { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 13, fontFace: "Calibri", align: "center" } },
  { text: "D4RL Score", options: { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 13, fontFace: "Calibri", align: "center" } },
];
let fmRows = [
  ["HF DT (Foundation, hopper-medium)", "69.6", "2.8"],
  ["BC (Ours)", "1,481.4", "46.1"],
  ["DT (Ours)", "968.9", "30.4"],
  ["ODT (Ours, online FT)", "1,849.2", "57.4"],
].map((row, ri) => row.map((cell, ci) => ({
  text: cell,
  options: {
    fontSize: 13, fontFace: "Calibri",
    fill: { color: ri === 0 ? "FFF3E0" : C.white },
    color: ci === 0 ? C.navy : (ri === 0 && ci === 2) ? C.red : "444444",
    bold: ci === 0 || ri === 0,
    align: ci === 0 ? "left" : "center",
  }
})));

s15.addTable([fmHeader, ...fmRows], {
  x: 0.5, y: 1.15, w: 9, colW: [4.5, 2.25, 2.25],
  border: { pt: 0.5, color: "DDDDDD" }, rowH: 0.42,
});

// Analysis card
s15.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 3.0, w: 9, h: 2.0, fill: { color: C.white }, shadow: makeShadow()
});
s15.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.0, w: 0.08, h: 2.0, fill: { color: C.orange } });
s15.addText("Why does the Foundation Model score only 2.8?", {
  x: 0.85, y: 3.1, w: 8.3, h: 0.35, fontSize: 16, fontFace: "Georgia", color: C.navy, bold: true
});
s15.addText([
  { text: "Dataset distribution mismatch: ", options: { bold: true } },
  { text: "Foundation model was trained on hopper-medium (consistent medium-quality data). Our environment uses the medium-replay distribution with very different state/return statistics.", options: { breakLine: true } },
  { text: "The pre-trained model's weights are calibrated for a different data regime — it literally doesn't understand the observation scale of our environment.", options: { breakLine: true } },
  { text: "This demonstrates that foundation models do NOT automatically transfer across dataset distributions, even within the same environment.", options: { bold: true } },
], { x: 0.85, y: 3.5, w: 8.3, h: 1.4, fontSize: 12, fontFace: "Calibri", color: "444444", paraSpaceAfter: 4 });

addFooter(s15, sn, TOTAL);
s15.addNotes(`This is one of our most interesting findings. The HuggingFace pre-trained Decision Transformer — a foundation model trained on hopper-medium — scored only 2.8 D4RL on our environment. That's barely above random!

Why? The key reason is dataset distribution mismatch. The foundation model was trained on hopper-medium-v2, which contains only medium-quality trajectories from a single partially trained policy. Our environment evaluates on the medium-replay distribution, which has very different state statistics — it includes everything from catastrophic failures to near-expert runs. The model's weights are calibrated for a completely different observation and return scale.

This is actually a very important finding for the field. It shows that even within the same MuJoCo environment, a pre-trained foundation model does NOT automatically transfer across dataset distributions. Domain-specific training on the target distribution — what we did — significantly outperforms a larger pre-trained model that was trained on different data. All three of our models crush the foundation model.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 16: Error & Failure Analysis
// ════════════════════════════════════════════════════════════════
sn++;
let s16 = pres.addSlide();
s16.background = { color: C.lightBg };
addSectionHeader(s16, "Error and Failure Analysis");

const failures = [
  {
    title: "DT Evaluation Instability",
    desc: "D4RL scores swing wildly (13 to 47) across evaluation checkpoints. Small weight changes cause large behavioral shifts in generated trajectories.",
    fix: "More evaluation episodes and longer training would stabilize. RTG sweep at test time partially mitigates this.",
    color: C.accent
  },
  {
    title: "ODT Offline Phase Plateau",
    desc: "ODT scores only ~14 D4RL during offline training. The entropy bonus keeps the policy too stochastic, preventing exploitation of good trajectories.",
    fix: "The online phase compensates — scores jump to 52+ after live fine-tuning, validating the two-phase design.",
    color: C.green
  },
  {
    title: "Improvement Techniques vs. Evaluation",
    desc: "All 3 techniques lowered validation loss but didn't improve D4RL scores. Offline supervised loss doesn't directly correlate with RL control quality.",
    fix: "This gap between loss and performance is a known challenge in offline RL — future work should optimize evaluation-aware objectives.",
    color: C.orange
  },
];

failures.forEach((f, i) => {
  const cy = 1.15 + i * 1.42;
  s16.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: cy, w: 9, h: 1.3, fill: { color: C.white }, shadow: makeShadow()
  });
  s16.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: cy, w: 0.08, h: 1.3, fill: { color: f.color } });
  s16.addText(f.title, {
    x: 0.85, y: cy + 0.08, w: 8.3, h: 0.3, fontSize: 15, fontFace: "Georgia", color: C.navy, bold: true, margin: 0
  });
  s16.addText(f.desc, {
    x: 0.85, y: cy + 0.45, w: 4.5, h: 0.75, fontSize: 11, fontFace: "Calibri", color: "555555"
  });
  s16.addText(f.fix, {
    x: 5.5, y: cy + 0.45, w: 3.7, h: 0.75, fontSize: 11, fontFace: "Calibri", color: C.navy, italic: true
  });
});

addFooter(s16, sn, TOTAL);
s16.addNotes(`Let me walk through our three main failure cases and what we learned from them.

First, DT evaluation instability. The D4RL scores swing wildly between 13 and 47 across different checkpoints. This is because in RL, small weight changes can cause large behavioral shifts — the model might suddenly fall over instead of hopping. More evaluation episodes and RTG sweeping help, but this remains a challenge.

Second, the ODT offline phase plateau. During pure offline training, ODT only scores about 14. This is actually by design — the entropy bonus deliberately keeps the policy stochastic so it learns diverse strategies. The payoff comes in the online phase where scores jump to 52 and above.

Third, the gap between loss and performance for our improvement techniques. All three techniques lowered validation loss — meaning they fit the offline data better — but didn't improve the actual robot control. This is a fundamental challenge in offline RL: fitting the dataset well doesn't mean you can control the robot well. The model needs to generalize to states it hasn't seen, and lower loss doesn't guarantee that.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 17: Team Collaboration & Individual Contributions
// ════════════════════════════════════════════════════════════════
sn++;
let s17 = pres.addSlide();
s17.background = { color: C.lightBg };
addSectionHeader(s17, "Team Collaboration & Contributions");

// Shared work
s17.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.2, w: 9, h: 1.0, fill: { color: C.navy }, shadow: makeShadow()
});
s17.addText("Shared Work", {
  x: 0.8, y: 1.25, w: 2, h: 0.35, fontSize: 14, fontFace: "Georgia", color: C.accent2, bold: true
});
s17.addText("Unified data processing pipeline, evaluation framework, and D4RL benchmarking infrastructure built collaboratively.", {
  x: 0.8, y: 1.65, w: 8.4, h: 0.45, fontSize: 12, fontFace: "Calibri", color: C.ice
});

// Individual cards
const members = [
  { name: "Tajaddin Gafarov", role: "Baseline Model (BC)", details: "Implemented Behavior Cloning transformer, training pipeline, performance improvement techniques analysis", color: C.accent },
  { name: "Nicholas Kovacs", role: "Decision Transformer", details: "Implemented deterministic DT with RTG conditioning, MSE training, return sweep evaluation", color: C.green },
  { name: "Vivekanandhan Kathirvel", role: "Online Decision Transformer", details: "Implemented stochastic ODT with entropy bonus, online fine-tuning phase, hindsight relabeling", color: C.orange },
];

members.forEach((m, i) => {
  const cx = 0.5 + i * 3.1;
  s17.addShape(pres.shapes.RECTANGLE, {
    x: cx, y: 2.55, w: 2.85, h: 2.5, fill: { color: C.white }, shadow: makeShadow()
  });
  s17.addShape(pres.shapes.RECTANGLE, { x: cx, y: 2.55, w: 2.85, h: 0.06, fill: { color: m.color } });
  s17.addText(m.name, {
    x: cx + 0.15, y: 2.7, w: 2.55, h: 0.35, fontSize: 14, fontFace: "Georgia", color: C.navy, bold: true
  });
  s17.addText(m.role, {
    x: cx + 0.15, y: 3.1, w: 2.55, h: 0.3, fontSize: 12, fontFace: "Calibri", color: m.color, bold: true
  });
  s17.addText(m.details, {
    x: cx + 0.15, y: 3.45, w: 2.55, h: 1.4, fontSize: 11, fontFace: "Calibri", color: "555555"
  });
});

addFooter(s17, sn, TOTAL);
s17.addNotes(`For team collaboration, we first built the shared infrastructure together — the data processing pipeline, evaluation framework, and D4RL benchmarking code. This ensured all three models were evaluated in the exact same way for fair comparison. Then we branched off individually. Tajaddin built the Behavior Cloning baseline and led the performance improvement techniques analysis. Nicholas implemented the deterministic Decision Transformer with return-to-go conditioning. And Vivek built the Online Decision Transformer with the stochastic policy, entropy bonus, and online fine-tuning phase. All three models share the same codebase structure and evaluation pipeline.`);

// ════════════════════════════════════════════════════════════════
// SLIDE 18: Conclusion
// ════════════════════════════════════════════════════════════════
sn++;
let s18 = pres.addSlide();
s18.background = { color: C.darkBg };
s18.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

s18.addText("Conclusion", {
  x: 0.8, y: 0.4, w: 8.4, h: 0.7, fontSize: 36, fontFace: "Georgia", color: C.white, bold: true
});
s18.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.1, w: 2, h: 0.04, fill: { color: C.accent } });

s18.addText([
  { text: "Key Results", options: { fontSize: 18, bold: true, color: C.accent2, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "ODT achieves best performance (63.8 D4RL) — online fine-tuning is critical", options: { bullet: true, breakLine: true } },
  { text: "BC is a surprisingly strong baseline (46.1) on mixed-quality data", options: { bullet: true, breakLine: true } },
  { text: "Foundation models don't transfer across dataset distributions (2.8 D4RL)", options: { bullet: true, breakLine: true } },
  { text: "Offline loss and RL evaluation measure fundamentally different things", options: { bullet: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 12 } },
  { text: "Future Work", options: { fontSize: 18, bold: true, color: C.accent2, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "Train longer with improvement techniques (especially K=30 context)", options: { bullet: true, breakLine: true } },
  { text: "Evaluation-aware training objectives instead of pure supervised loss", options: { bullet: true, breakLine: true } },
  { text: "Foundation model fine-tuning on target distribution", options: { bullet: true } },
], { x: 0.8, y: 1.35, w: 8.4, h: 3.5, fontSize: 14, fontFace: "Calibri", color: C.ice, paraSpaceAfter: 4 });

s18.addText("Thank you — Questions?", {
  x: 0.8, y: 4.85, w: 8.4, h: 0.5, fontSize: 20, fontFace: "Georgia", color: C.accent2, italic: true
});

s18.addNotes(`To conclude, our key results: ODT achieves the best performance at 63.8 D4RL score, proving that online fine-tuning is critical for offline RL with transformers. BC is a surprisingly strong baseline at 46.1 — simple imitation learning works well when the dataset contains good trajectories. The foundation model comparison revealed that pre-trained models don't automatically transfer across dataset distributions, scoring only 2.8. And our improvement technique experiments showed that offline loss and RL evaluation measure fundamentally different things.

For future work, we'd train longer with the improvement techniques — especially the K=30 context length which showed the lowest validation loss but needs more epochs. We'd also explore evaluation-aware training objectives and fine-tuning the foundation model on our target distribution.

Thank you. We're happy to take any questions.`);

// ── Write file ──
const outputPath = path.resolve("G:/Project 2/Main/outputs/Final_Presentation.pptx");
pres.writeFile({ fileName: outputPath }).then(() => {
  console.log("Presentation saved to: " + outputPath);
}).catch(err => {
  console.error("Error:", err);
});
