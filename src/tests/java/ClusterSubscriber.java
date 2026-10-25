import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.PubSubOption;
import edu.wpi.first.util.struct.StructBuffer;
import java.nio.charset.StandardCharsets;

/** Standalone Java WPILib NT4 server; no robot project or C++ decoder involved. */
public final class ClusterSubscriber {
  private static void require(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }

  private static void near(double actual, double expected, String field) {
    require(Double.isFinite(actual) && Math.abs(actual - expected) < 1e-7,
        field + ": expected " + expected + ", received " + actual);
  }

  public static void main(String[] args) throws Exception {
    int port = args.length == 0 ? 15810 : Integer.parseInt(args[0]);
    try (var instance = NetworkTableInstance.create();
        var clusters = instance.getStructArrayTopic("/COS/Clusters", ClusterNT.struct)
            .subscribe(new ClusterNT[0], PubSubOption.periodic(0.001),
                PubSubOption.sendAll(true), PubSubOption.keepDuplicates(true),
                PubSubOption.pollStorage(4096));
        // WPILib 2026.2.1's typed readQueue() discards empty arrays and corrupts
        // mixed empty/nonempty queues. Keep the exact struct type on the wire
        // and let WPILib's StructBuffer decode every raw queue item instead.
        var wire = instance.getRawTopic("/COS/Clusters")
            .subscribe("struct:ClusterNT[]", new byte[0], PubSubOption.periodic(0.001),
                PubSubOption.sendAll(true), PubSubOption.keepDuplicates(true),
                PubSubOption.pollStorage(4096));
        var schema = instance.getRawTopic("/.schema/struct:ClusterNT")
            .subscribe("structschema", new byte[0]);
        var ready = instance.getBooleanTopic("/cluster-test/ready").publish();
        var done = instance.getBooleanTopic("/cluster-test/done").subscribe(false);
        var ack = instance.getBooleanTopic("/cluster-test/ack").publish()) {
      instance.startServer("", "", 0, port);
      ready.set(true);
      System.out.println("JAVA_READY port=" + port);
      System.out.flush();
      int received = 0, batches = 0, empties = 0, large = 0;
      var buffer = StructBuffer.create(ClusterNT.struct);
      long previousTimestamp = 0, firstArrival = 0, lastArrival = 0;
      long deadline = System.nanoTime() + 40_000_000_000L;
      while (received < 1000 || !done.get()) {
        require(System.nanoTime() < deadline,
            "Timed out after receiving " + received + "/1000 samples");
        var queue = wire.readQueue();
        if (queue.length > 0) {
          lastArrival = System.nanoTime();
          if (firstArrival == 0) firstArrival = lastArrival;
          batches++;
        }
        for (var sample : queue) {
          var values = buffer.readArray(sample.value);
          require(received < 1000, "Extra sample");
          require(sample.timestamp > previousTimestamp, "Non-increasing timestamp");
          previousTimestamp = sample.timestamp;
          int sequence = received / 2;
          int count = sequence % 40 == 0 ? 0 : sequence % 40 == 1 ? 128 : 2;
          require(values.length == count,
              "Array length mismatch at frame " + received);
          if (count == 0) empties++;
          if (count == 128) large++;
          for (int index = 0; index < count; index++) {
            var cluster = values[index];
            near(cluster.centerX(), 12.5 + index * 0.25, "center_x");
            near(cluster.centerY(), -2.75 + sequence + index * 0.125, "center_y");
            near(cluster.fieldRelativeSemimajorAxisYaw(), Math.PI / 2, "yaw");
            near(cluster.semimajorLength(), 6, "semimajor_length");
            near(cluster.semiminorLength(), 2, "semiminor_length");
            near(cluster.predictedDensity(), (sequence % 17) * 0.25 + index + 0.5,
                "predicted_density");
          }
          received++;
        }
        Thread.sleep(0, 500_000);
      }
      require(clusters.getTopic().getTypeString().equals("struct:ClusterNT[]"),
          "Incorrect topic type");
      var typedLatest = clusters.get();
      require(typedLatest.length == 2, "Typed Java subscriber could not decode the last array");
      near(typedLatest[0].centerY(), 496.25, "typed subscriber center_y");
      require(new String(schema.get(), StandardCharsets.UTF_8).equals(ClusterNT.struct.getSchema()),
          "C++ schema does not match the Java contract");
      double seconds = (lastArrival - firstArrival) / 1e9;
      double sampleHz = (received - 1) / seconds;
      double batchHz = (batches - 1) / seconds;
      require(sampleHz > 150, "Sample delivery too slow: " + sampleHz + " Hz");
      require(batchHz > 110, "Network arrival batches did not exceed 100 Hz: " + batchHz);
      // Keep the publisher connected long enough to detect unexpected duplicates.
      Thread.sleep(200);
      require(wire.readQueue().length == 0, "Unexpected extra samples");
      ack.set(true);
      instance.flush();
      Thread.sleep(300);
      System.out.printf(
          "PASS arrays=%d empty=%d large=%d sample_hz=%.1f arrival_batch_hz=%.1f%n",
          received, empties, large, sampleHz, batchHz);
    }
  }
}
