import tensorflow as tf

box = tf.constant([
    [0, 0, 1, 2],
    [0, 1, 1, 3],
    [2, 3, 2, 2]
])
scores = tf.constant([
    [0.45, 0.3, 1, 2],
    [0.3, 1, 1, 0.2],
    [2, 0.46, 2, 2]
])
without_match = tf.constant([0, 3])


mask = box[..., None] == without_match[None, None]
box_ind = tf.argmax(mask, axis=-1)[..., None]
scores = tf.where(tf.reduce_any(mask, axis=-1), scores, -100)

# keep shape so we can undo the flattening
shape = tf.shape(box)
width = shape[1]
height = shape[0]
xs = tf.range(width, dtype=tf.int64)
ys = tf.range(height, dtype=tf.int64)
grid_loc = tf.stack(tf.meshgrid(xs, ys), axis=-1)[..., ::-1]

indices = tf.concat([grid_loc, box_ind], axis=-1)
scores_per_box = tf.scatter_nd(indices, scores, (shape[0], shape[1], 2))
# print(scores_per_box.numpy())
print(scores_per_box.numpy()[:, :, 1])
max_vals = tf.reduce_max(scores_per_box, [0, 1])
tf.where(scores >)
print(max_vals.numpy())
