import edu.wpi.first.util.struct.Struct;
import edu.wpi.first.util.struct.StructSerializable;
import java.nio.ByteBuffer;

/** Wire contract shared with gamepiece/cluster_nt.h. Units: m, rad, m, balls/m². */
public record ClusterNT(
    double centerX,
    double centerY,
    double fieldRelativeSemimajorAxisYaw,
    double semimajorLength,
    double semiminorLength,
    double predictedDensity) implements StructSerializable {
  public static final Struct<ClusterNT> struct = new Struct<>() {
    @Override
    public Class<ClusterNT> getTypeClass() { return ClusterNT.class; }

    @Override
    public String getTypeName() { return "ClusterNT"; }

    @Override
    public int getSize() { return 6 * kSizeDouble; }

    @Override
    public String getSchema() {
      return "double center_x;double center_y;"
          + "double field_relative_semimajor_axis_yaw;double semimajor_length;"
          + "double semiminor_length;double predicted_density";
    }

    @Override
    public ClusterNT unpack(ByteBuffer buffer) {
      return new ClusterNT(buffer.getDouble(), buffer.getDouble(), buffer.getDouble(),
          buffer.getDouble(), buffer.getDouble(), buffer.getDouble());
    }

    @Override
    public void pack(ByteBuffer buffer, ClusterNT value) {
      buffer.putDouble(value.centerX());
      buffer.putDouble(value.centerY());
      buffer.putDouble(value.fieldRelativeSemimajorAxisYaw());
      buffer.putDouble(value.semimajorLength());
      buffer.putDouble(value.semiminorLength());
      buffer.putDouble(value.predictedDensity());
    }

    @Override
    public boolean isImmutable() { return true; }
  };
}
