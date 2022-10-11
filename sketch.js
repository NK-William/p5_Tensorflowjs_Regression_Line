let x_vals = [];
let y_vals = [];

let m, b;

// Training meaning: minimize the loss() function with the optimizer, adjusting m and b based on that.
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function loss(pred, labels) {
  // pred are y values(in tensors) from predict function.
  // labels(in tensors) are the actual y values.
   return pred.sub(labels).square().mean()
  };

function setup() {
  createCanvas(400, 400);

  m = tf.variable(tf.scalar(random(1))); // don't clear the variable from the memory(this is what we are adjusting during training).
  b = tf.variable(tf.scalar(random(1))); // don't clear the variable from the memory(this is what we are adjusting during training).
}

function predict(x_vals){
  const tfxs = tf.tensor1d(x_vals);

  // y = mx + b;
  const Pred_tf_ys = tfxs.mul(m).add(b);

  return Pred_tf_ys;
}

function mousePressed(){
  // setting scalers -> x-axis will range from 0 - 1 and  y-axis will also range from 0 - 1.
 let x = map(mouseX, 0, width, 0, 1);
 let y = map(mouseY, 0, height, 1, 0);

  x_vals.push(x);
  y_vals.push(y);
}

function draw(){

  tf.tidy(() => {
  if(x_vals.length > 0){
    const label_tf_ys = tf.tensor1d(y_vals);
    // train the model
  // minimize (f, returnCost?, varList?)
    // Note: Here the "varList?" in the parameter are not specified, so it defaults to all trainable variables(m and b(tf.variable))
    // since list is not provided.
  optimizer.minimize(() => loss(predict(x_vals), label_tf_ys));
  }
});
  
  background(0);

  stroke(255);
  strokeWeight(8);
  for(let i = 0; i < x_vals.length; i++){
    let px = map(x_vals[i], 0, 1, 0, width);
    let py = map(y_vals[i], 0, 1, height, 0);
    point(px, py);
  }

 
  const lineX = [0, 1];
  const ys = tf.tidy(() => predict(lineX));
  let lineY = ys.dataSync();
  ys.dispose();
  // ys.print();

  let x1 = map(lineX[0], 0, 1, 0, width);
  let x2 = map(lineX[1], 0, 1, 0, width);

  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);

  strokeWeight(2);
  line(x1, y1, x2, y2);
}

