let x_vals = [];
let y_vals = [];

let a, b, c, d;
let dragging = false;

// Training meaning: minimize the loss() function with the optimizer, adjusting m and b based on that.
const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);

function loss(pred, labels) {
  // pred are y values(in tensors) from predict function.
  // labels(in tensors) are the actual y values.
   return pred.sub(labels).square().mean()
  };

function setup() {
  createCanvas(400, 400);

  a = tf.variable(tf.scalar(random(-1, 1))); // don't clear the variable from the memory(this is what we are adjusting during training).
  b = tf.variable(tf.scalar(random(-1, 1))); // don't clear the variable from the memory(this is what we are adjusting during training).
  c = tf.variable(tf.scalar(random(-1, 1))); // don't clear the variable from the memory(this is what we are adjusting during training).
  d = tf.variable(tf.scalar(random(-1, 1))); // don't clear the variable from the memory(this is what we are adjusting during training).
}

function predict(x_vals){
  const tfxs = tf.tensor1d(x_vals);

  /*// y = mx + b;
  const Pred_tf_ys = tfxs.mul(m).add(b);*/

 // y = ax^3 + bx^2 + cx + d
 const Pred_tf_ys = tfxs.pow(tf.scalar(3)).mul(a)
        .add(tfxs.square().mul(b))
        .add(tfxs.mul(c))
        .add(d);

  return Pred_tf_ys;
}

function mousePressed(){
  dragging = true;
}

function mouseReleased(){
  dragging = false;
}

/* function mouseDragged(){ 
 
}
 */
function draw(){

  if(dragging){
    // setting scalers -> x-axis will range from -1 to 1 and  y-axis will also range from -1 to 1.
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);

    x_vals.push(x);
    y_vals.push(y);
  }else{
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
  }

  
  
  background(0);

  stroke(255);
  strokeWeight(8);
  for(let i = 0; i < x_vals.length; i++){
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, height, 0);
    point(px, py);
  }

 
  const curveX = [];
  for(let x = -1; x < 1.01; x += 0.05){
    curveX.push(x)
  }
  


  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();
  
  // draw a quadratic curve
  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for(let i = 0; i < curveX.length; i++){
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, height, 0);
    vertex(x, y);
  }

  endShape();

  /* this was working for linear regressio
  let x1 = map(lineX[0], -1, 1, 0, width);
  let x2 = map(lineX[1], -1, 1, 0, width);

  let y1 = map(lineY[0], -1, 1, height, 0);
  let y2 = map(lineY[1], -1, 1, height, 0);

  strokeWeight(2);
  line(x1, y1, x2, y2);*/
}

