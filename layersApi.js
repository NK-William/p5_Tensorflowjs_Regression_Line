const model = tf.sequential();

const hidden = tf.layers.dense();
const output = tf.layers.dense();

model.addLayer(hidden);
model.addLayer(output);
