module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func @main(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Squeeze"(%arg0) {T = f32, device = "", squeeze_dims = [-3]} : (tensor<?x?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %1 : tensor <?x?xf32>
    }
    return %graph : tensor <?x?xf32>
  }
}
