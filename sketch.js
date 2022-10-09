let xs = [];
let ys = [];

let m, b;

// Training meaning: minimize the loss() function with the optimizer, adjusting m and b based on that.
const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

function loss(pred, labels) {
  // pred are y values from predict function.
  // labels are the actual y values.
   return pred.sub(labels).square().mean()
  };

function setup() {
  createCanvas(400, 400);

  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function predict(xs){
  const tfxs = tf.tensor1d(xs);

  // y = mx + b;
  const ys = tfxs.mul(m).add(b);

  return ys;
}

function mousePressed(){
  // setting scalers -> x-axis will range from 0 - 1 and  y-axis will also range from 0 - 1.
 let x = map(mouseX, 0, width, 0, 1);
 let y = map(mouseY, 0, height, 1, 0);

  xs.push(x);
  ys.push(y);
}

function draw(){

  background(0);

  stroke(255);
  strokeWeight(8);
  for(let i = 0; i < xs.length; i++){
    let px = map(xs[i], 0, 1, 0, width);
    let py = map(ys[i], 0, 1, height, 0);
    point(px, py);
  }
}

