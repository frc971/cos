#include "tools/path_camera_sim/field_renderer.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace path_camera_sim {
namespace {

using Json = nlohmann::json;
using Matrix = Eigen::Matrix4f;

auto ReadJson(const std::filesystem::path& path) -> Json {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open " + path.string());
  return Json::parse(input);
}

auto ReadGlb(const std::filesystem::path& path)
    -> std::pair<Json, std::vector<unsigned char>> {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open " + path.string());
  const std::vector<unsigned char> bytes{
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  auto word = [&](size_t offset) -> uint32_t {
    if (offset + 4 > bytes.size()) throw std::runtime_error("Truncated GLB");
    uint32_t result;
    std::memcpy(&result, bytes.data() + offset, 4);
    return result;
  };
  if (bytes.size() < 28 || word(0) != 0x46546c67 || word(4) != 2 ||
      word(8) != bytes.size()) {
    throw std::runtime_error("Invalid GLB: " + path.string());
  }
  Json document;
  std::vector<unsigned char> binary;
  for (size_t offset = 12; offset + 8 <= bytes.size();) {
    const size_t length = word(offset);
    const uint32_t kind = word(offset + 4);
    offset += 8;
    if (length > bytes.size() - offset) throw std::runtime_error("Truncated GLB chunk");
    if (kind == 0x4e4f534a) {
      document = Json::parse(bytes.begin() + offset, bytes.begin() + offset + length);
    } else if (kind == 0x004e4942) {
      binary.assign(bytes.begin() + offset, bytes.begin() + offset + length);
    }
    offset += length;
  }
  if (document.is_null() || binary.empty()) throw std::runtime_error("GLB has no mesh data");
  return {std::move(document), std::move(binary)};
}

auto NodeTransform(const Json& node) -> Matrix {
  Matrix result = Matrix::Identity();
  if (node.contains("matrix")) {
    for (int col = 0; col < 4; ++col) {
      for (int row = 0; row < 4; ++row) {
        result(row, col) = node.at("matrix").at(col * 4 + row).get<float>();
      }
    }
    return result;
  }
  if (node.contains("translation")) {
    for (int i = 0; i < 3; ++i) {
      result(i, 3) = node.at("translation").at(i).get<float>();
    }
  }
  if (node.contains("rotation")) {
    const auto& q = node.at("rotation");
    result.block<3, 3>(0, 0) = Eigen::Quaternionf(
        q.at(3).get<float>(), q.at(0).get<float>(),
        q.at(1).get<float>(), q.at(2).get<float>()).toRotationMatrix();
  }
  if (node.contains("scale")) {
    for (int i = 0; i < 3; ++i) {
      result.block<3, 1>(0, i) *= node.at("scale").at(i).get<float>();
    }
  }
  return result;
}

auto MakeShader(GLenum kind, const char* source) -> GLuint {
  const GLuint shader = glCreateShader(kind);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint ok = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char message[1024] = {};
    glGetShaderInfoLog(shader, sizeof(message), nullptr, message);
    throw std::runtime_error(std::string("GL shader compilation failed: ") + message);
  }
  return shader;
}

auto MakeProgram() -> GLuint {
  constexpr char vertex[] = R"(#version 300 es
    layout(location=0) in vec3 position;
    layout(location=1) in vec3 normal;
    uniform mat4 mvp;
    uniform mat4 model;
    out float light;
    out vec3 surface;
    out vec3 world_position;
    out vec3 world_normal;
    void main() {
      gl_Position = mvp * vec4(position, 1.0);
      gl_PointSize = 1.0;
      vec3 n = normalize(mat3(model) * normal);
      surface = normal;
      world_position = (model * vec4(position, 1.0)).xyz;
      world_normal = n;
      light = 0.55 + 0.45 * abs(dot(n, normalize(vec3(-0.4, 0.3, 1.0))));
    }
  )";
  constexpr char fragment[] = R"(#version 300 es
    precision highp float;
    uniform vec4 color;
    uniform bool fuel_labels;
    uniform float soot;
    uniform float appearance_seed;
    uniform int material;
    uniform sampler2D room_texture;
    in float light;
    in vec3 surface;
    in vec3 world_position;
    in vec3 world_normal;
    out vec4 pixel;
    float hash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
    float noise(vec3 p) {
      vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
      return mix(mix(mix(hash(i), hash(i+vec3(1,0,0)), f.x),
                     mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),
                 mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x),
                     mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y), f.z);
    }
    void main() {
      if (fuel_labels) { pixel = color; return; }
      vec3 base = color.rgb;
      if (material > 0) {
        vec2 uv = material == 1 ? world_position.xy / vec2(0.6, 0.15) :
            vec2(abs(world_normal.x) > 0.5 ? world_position.y : world_position.x, world_position.z) / vec2(1.2, 0.9);
        vec3 tex = texture(room_texture, uv).rgb;
        float grain = material == 1 ? 0.92 + 0.16 * noise(world_position * 1200.0) : 1.0;
        float seam = material == 2 && (fract(uv.x * 0.5) < 0.006 || fract(uv.y * 0.75) < 0.006) ? 0.65 : 1.0;
        base *= pow(tex, vec3(2.2)) * grain * seam;
      }
      if (soot > 0.0) {
        vec3 p = normalize(surface) * vec3(7.0, 9.0, 12.0) + appearance_seed;
        float smudge = smoothstep(0.38, 0.76, 0.7 * noise(p) + 0.3 * noise(p * 2.7));
        float speckle = smoothstep(0.68, 0.86, noise(p * 7.0));
        float dirt = soot * (0.75 * smudge + 0.25 * speckle);
        base = mix(base, vec3(0.055, 0.043, 0.026), dirt);
        base *= 1.0 - soot * 0.18 * noise(p * 1.7);
      }
      pixel = vec4(pow(max(base * light, vec3(0.0)), vec3(1.0 / 2.2)), color.a);
    }
  )";
  const GLuint v = MakeShader(GL_VERTEX_SHADER, vertex);
  const GLuint f = MakeShader(GL_FRAGMENT_SHADER, fragment);
  const GLuint program = glCreateProgram();
  glAttachShader(program, v);
  glAttachShader(program, f);
  glLinkProgram(program);
  glDeleteShader(v);
  glDeleteShader(f);
  GLint ok = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &ok);
  if (!ok) {
    char message[1024] = {};
    glGetProgramInfoLog(program, sizeof(message), nullptr, message);
    throw std::runtime_error(std::string("GL program link failed: ") + message);
  }
  return program;
}

}  // namespace

FieldRenderer::FieldRenderer(const std::filesystem::path& field_directory,
                             const cv::Size& image_size,
                             const cv::Matx33d& intrinsics)
    : image_size_(image_size), projection_(Matrix::Zero()) {
  const Json config = ReadJson(field_directory / "config.json");
  const auto [scene, binary] = ReadGlb(field_directory / "model.glb");
  const float length = config.at("widthInches").get<float>() * 0.0254f;
  const float width = config.at("heightInches").get<float>() * 0.0254f;
  field_length_ = length;
  field_width_ = width;
  Matrix field_to_world = Matrix::Identity();
  field_to_world(0, 0) = -1;
  field_to_world(1, 1) = -1;
  field_to_world(0, 3) = length / 2;
  field_to_world(1, 3) = width / 2;
  for (const auto& rotation : config.at("rotations")) {
    Eigen::Vector3f axis;
    const std::string name = rotation.at("axis").get<std::string>();
    if (name == "x") axis = Eigen::Vector3f::UnitX();
    else if (name == "y") axis = Eigen::Vector3f::UnitY();
    else if (name == "z") axis = Eigen::Vector3f::UnitZ();
    else throw std::runtime_error("Unknown field model rotation axis");
    Matrix turn = Matrix::Identity();
    turn.block<3, 3>(0, 0) = Eigen::AngleAxisf(
        rotation.at("degrees").get<float>() * static_cast<float>(M_PI / 180.0),
        axis).toRotationMatrix();
    field_to_world *= turn;
  }

  const auto& nodes = scene.at("nodes");
  const auto& meshes = scene.at("meshes");
  const auto& accessors = scene.at("accessors");
  const auto& views = scene.at("bufferViews");
  const auto& materials = scene.at("materials");
  auto offset = [&](const Json& accessor) -> size_t {
    const auto& view = views.at(accessor.at("bufferView").get<size_t>());
    return view.at("byteOffset").get<size_t>() + accessor.value("byteOffset", 0U);
  };
  std::function<void(size_t, const Matrix&)> visit = [&](size_t index, const Matrix& parent) {
    const auto& node = nodes.at(index);
    const bool is_fuel = node.value("name", std::string{}).starts_with("GE-26900: Fuel");
    const Matrix model = parent * NodeTransform(node);
    if (node.contains("mesh")) {
      for (const auto& primitive : meshes.at(node.at("mesh").get<size_t>()).at("primitives")) {
        if (primitive.value("mode", 4) != 4) throw std::runtime_error("Expected GLB triangles");
        const auto& position = accessors.at(primitive.at("attributes").at("POSITION").get<size_t>());
        const auto& normal = accessors.at(primitive.at("attributes").at("NORMAL").get<size_t>());
        const auto& indices = accessors.at(primitive.at("indices").get<size_t>());
        const auto& view = views.at(position.at("bufferView").get<size_t>());
        const auto& material = materials.at(primitive.value("material", 0U));
        const auto& base = material.at("pbrMetallicRoughness").at("baseColorFactor");
        Draw draw{.model = model,
                  .color = cv::Vec4f(base.at(0).get<float>(), base.at(1).get<float>(),
                                    base.at(2).get<float>(), base.at(3).get<float>()),
                  .position_offset = static_cast<GLint>(offset(position)),
                  .normal_offset = static_cast<GLint>(offset(normal)),
                  .vertex_stride = static_cast<GLsizei>(view.value("byteStride", 12)),
                  .index_count = static_cast<GLsizei>(indices.at("count").get<size_t>()),
                  .index_type = indices.at("componentType").get<GLenum>(),
                  .index_offset = offset(indices),
                  .transparent = material.value("alphaMode", std::string("OPAQUE")) == "BLEND"};
        if (draw.index_type != GL_UNSIGNED_SHORT && draw.index_type != GL_UNSIGNED_INT)
          throw std::runtime_error("Unsupported GLB index type");
        if (is_fuel) staged_fuel_draws_.push_back(std::move(draw));
        else draws_.push_back(std::move(draw));
      }
    }
    for (const auto& child : node.value("children", Json::array())) {
      visit(child.get<size_t>(), model);
    }
  };
  const size_t scene_index = scene.value("scene", 0U);
  for (const auto& root : scene.at("scenes").at(scene_index).at("nodes")) {
    visit(root.get<size_t>(), field_to_world);
  }
  if (draws_.empty()) throw std::runtime_error("Field GLB has no meshes");
  const size_t expected_fuel = config.at("gamePieces").at(0).at("stagedObjects").size();
  if (staged_fuel_draws_.size() != expected_fuel)
    throw std::runtime_error("Staged Fuel mesh count does not match field config");

  const float w = image_size.width;
  const float h = image_size.height;
  constexpr float near = 0.02f;
  constexpr float far = 100.0f;
  projection_(0, 0) = 2 * intrinsics(0, 0) / w;
  projection_(0, 2) = 2 * intrinsics(0, 2) / w - 1;
  projection_(1, 1) = -2 * intrinsics(1, 1) / h;
  projection_(1, 2) = 1 - 2 * intrinsics(1, 2) / h;
  projection_(2, 2) = (far + near) / (far - near);
  projection_(2, 3) = -2 * far * near / (far - near);
  projection_(3, 2) = 1;

  display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (display_ == EGL_NO_DISPLAY || !eglInitialize(display_, nullptr, nullptr))
    throw std::runtime_error("Cannot initialize EGL display");
  if (!eglBindAPI(EGL_OPENGL_ES_API)) throw std::runtime_error("Cannot bind OpenGL ES");
  const EGLint attributes[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                               EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                               EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
                               EGL_BLUE_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_NONE};
  EGLConfig selected;
  EGLint count = 0;
  if (!eglChooseConfig(display_, attributes, &selected, 1, &count) || count == 0)
    throw std::runtime_error("Cannot choose EGL config");
  const EGLint surface_attributes[] = {EGL_WIDTH, image_size.width,
                                       EGL_HEIGHT, image_size.height, EGL_NONE};
  surface_ = eglCreatePbufferSurface(display_, selected, surface_attributes);
  const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
  context_ = eglCreateContext(display_, selected, EGL_NO_CONTEXT, context_attributes);
  if (surface_ == EGL_NO_SURFACE || context_ == EGL_NO_CONTEXT ||
      !eglMakeCurrent(display_, surface_, surface_, context_))
    throw std::runtime_error("Cannot create OpenGL ES 3 context");

  program_ = MakeProgram();
  glGenBuffers(1, &vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  glBufferData(GL_ARRAY_BUFFER, binary.size(), binary.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_buffer_);
  glGenFramebuffers(1, &frame_buffer_);
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
  glGenTextures(1, &color_buffer_);
  glBindTexture(GL_TEXTURE_2D, color_buffer_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image_size.width, image_size.height,
               0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         color_buffer_, 0);
  glGenRenderbuffers(1, &depth_buffer_);
  glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                        image_size.width, image_size.height);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, depth_buffer_);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    throw std::runtime_error("Cannot create GL framebuffer");
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glViewport(0, 0, image_size.width, image_size.height);
}

void FieldRenderer::SetFuelLayout(double entropy, uint32_t seed) {
  if (!std::isfinite(entropy) || entropy < 0.0 || entropy > 1.0)
    throw std::runtime_error("Fuel entropy must be between 0 and 1");
  active_fuel_draws_.clear();
  fuel_positions_.clear();
  fuel_cluster_ids_.clear();
  std::mt19937 random(seed);
  std::uniform_real_distribution<double> removal_score(0.0, 1.0);
  std::normal_distribution<double> displacement(0.0, 2.0);
  for (const Draw& staged : staged_fuel_draws_) {
    const double score = removal_score(random);
    const double offset_x = displacement(random);
    const double offset_y = displacement(random);
    if (score < entropy * 0.5) continue;
    Draw fuel = staged;
    if (entropy > 0.0) {
      fuel.model(0, 3) = std::clamp(
          static_cast<double>(fuel.model(0, 3)) + entropy * offset_x,
          0.075, static_cast<double>(field_length_) - 0.075);
      fuel.model(1, 3) = std::clamp(
          static_cast<double>(fuel.model(1, 3)) + entropy * offset_y,
          0.075, static_cast<double>(field_width_) - 0.075);
    }
    fuel_positions_.push_back({fuel.model(0, 3), fuel.model(1, 3), fuel.model(2, 3)});
    active_fuel_draws_.push_back(std::move(fuel));
  }
}

void FieldRenderer::SetPracticeRoom(const std::filesystem::path& path) {
  const auto config = ReadJson(path);
  const float length = config.at("size_m").at(0), width = config.at("size_m").at(1);
  const float height = config.at("wall_height_m");
  if (!(length > 3 && length < 30 && width > 3 && width < 20 && height > .3 && height < 4))
    throw std::runtime_error("Invalid practice-room dimensions");
  field_length_ = length; field_width_ = width;
  glDeleteTextures(2, room_textures_.data()); glGenTextures(2, room_textures_.data());
  for (int i = 0; i < 2; ++i) {
    auto image = cv::imread((path.parent_path() / config.at(i == 0 ? "carpet_texture" : "plywood_texture").get<std::string>()).string());
    if (image.empty()) throw std::runtime_error("Cannot load practice-room texture");
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    glBindTexture(GL_TEXTURE_2D, room_textures_[i]);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, image.cols, image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
  }
  glDeleteBuffers(1, &room_buffer_); glGenBuffers(1, &room_buffer_);
  std::vector<unsigned char> bytes;
  draws_.clear();
  auto quad = [&](std::array<Eigen::Vector3f, 4> vertices, Eigen::Vector3f normal, int material) {
    const auto offset = bytes.size();
    for (const auto& vertex : vertices) for (const auto& v : {vertex, normal}) {
      const auto* p = reinterpret_cast<const unsigned char*>(v.data()); bytes.insert(bytes.end(), p, p + 3 * sizeof(float));
    }
    const auto indices_offset = bytes.size();
    const uint32_t indices[] = {0,1,2,0,2,3};
    const auto* p = reinterpret_cast<const unsigned char*>(indices); bytes.insert(bytes.end(), p, p + sizeof(indices));
    const float tint = config.value(material == 0 ? "ceiling_gain" : material == 1 ? "carpet_gain" : "plywood_gain", material == 0 ? .7f : 1.f);
    draws_.push_back(Draw{.model = Matrix::Identity(), .color = cv::Vec4f(tint,tint,tint,1),
        .position_offset = static_cast<GLint>(offset), .normal_offset = static_cast<GLint>(offset + 12),
        .vertex_stride = 24, .index_count = 6, .index_type = GL_UNSIGNED_INT,
        .index_offset = indices_offset, .transparent = false, .buffer = room_buffer_, .material = material});
  };
  quad({Eigen::Vector3f{0,0,0}, {length,0,0}, {length,width,0}, {0,width,0}}, {0,0,1}, 1);
  auto wall = [&](float x1, float y1, float x2, float y2, float z) {
    const Eigen::Vector3f normal = Eigen::Vector3f(y2-y1, x1-x2, 0).normalized();
    quad({Eigen::Vector3f{x1,y1,0}, {x2,y2,0}, {x2,y2,z}, {x1,y1,z}}, normal, 2);
  };
  wall(0,0,length,0,height); wall(length,0,length,width,height);
  wall(length,width,0,width,height); wall(0,width,0,0,height);
  const float ceiling = 3.5f;
  quad({Eigen::Vector3f{0,0,ceiling}, {length,0,ceiling}, {length,width,ceiling}, {0,width,ceiling}}, {0,0,-1}, 0);
  quad({Eigen::Vector3f{0,0,height}, {length,0,height}, {length,0,ceiling}, {0,0,ceiling}}, {0,1,0}, 0);
  quad({Eigen::Vector3f{length,0,height}, {length,width,height}, {length,width,ceiling}, {length,0,ceiling}}, {-1,0,0}, 0);
  quad({Eigen::Vector3f{length,width,height}, {0,width,height}, {0,width,ceiling}, {length,width,ceiling}}, {0,-1,0}, 0);
  quad({Eigen::Vector3f{0,width,height}, {0,0,height}, {0,0,ceiling}, {0,width,ceiling}}, {1,0,0}, 0);
  for (const auto& p : config.value("partitions", Json::array())) wall(p.at(0),p.at(1),p.at(2),p.at(3),p.at(4));
  glBindBuffer(GL_ARRAY_BUFFER, room_buffer_);
  glBufferData(GL_ARRAY_BUFFER, bytes.size(), bytes.data(), GL_STATIC_DRAW);
}

void FieldRenderer::SetFuelClusters(const std::vector<FuelCluster>& clusters,
                                   double noise_m, uint32_t seed, double separation_m) {
  if (!std::isfinite(noise_m) || noise_m < 0.0)
    throw std::runtime_error("--fuel_noise must be finite and nonnegative");
  active_fuel_draws_.clear();
  fuel_positions_.clear();
  fuel_cluster_ids_.clear();
  std::mt19937 random(seed);
  std::normal_distribution<double> normal;
  for (std::size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
    const auto& cluster = clusters[cluster_id];
    const double sx = std::sqrt(cluster.covariance_m2[0]);
    const double sy = std::sqrt(cluster.covariance_m2[1]);
    const double count = std::round(2.0 * std::numbers::pi * cluster.density * sx * sy);
    if (!std::isfinite(count) || count > 100000 - fuel_positions_.size())
      throw std::runtime_error("Fuel clusters exceed the 100000-ball limit");
    if (count > 0 && staged_fuel_draws_.empty())
      throw std::runtime_error("Field model has no staged Fuel mesh");
    const double c = std::cos(cluster.rotation_rad);
    const double s = std::sin(cluster.rotation_rad);
    for (size_t i = 0; i < static_cast<size_t>(count); ++i) {
      bool placed = false;
      for (int attempt = 0; attempt < (separation_m > 0 ? 2000 : 1); ++attempt) {
        const double u = sx * normal(random);
        const double v = sy * normal(random);
        const double x = cluster.center_m[0] + c * u - s * v + noise_m * normal(random);
        const double y = cluster.center_m[1] + s * u + c * v + noise_m * normal(random);
        if (x < 0.075 || x > field_length_ - 0.075 ||
            y < 0.075 || y > field_width_ - 0.075) continue;
        if (separation_m > 0 && std::any_of(fuel_positions_.begin(), fuel_positions_.end(), [&](const auto& p) {
              return std::hypot(x - p[0], y - p[1]) < separation_m;
            })) continue;
        Draw fuel = staged_fuel_draws_.front();
        fuel.model(0, 3) = x;
        fuel.model(1, 3) = y;
        fuel_positions_.push_back({x, y, fuel.model(2, 3)});
        fuel_cluster_ids_.push_back(cluster_id);
        active_fuel_draws_.push_back(std::move(fuel));
        placed = true;
        break;
      }
      if (!placed && separation_m > 0) throw std::runtime_error("Cannot pack nonoverlapping Fuel cluster");
    }
  }
}

auto FieldRenderer::FuelPositions() const -> const std::vector<std::array<double, 3>>& {
  return fuel_positions_;
}

FieldRenderer::~FieldRenderer() {
  if (display_ == EGL_NO_DISPLAY) return;
  if (context_ != EGL_NO_CONTEXT) {
    glDeleteProgram(program_);
    glDeleteBuffers(1, &vertex_buffer_);
    glDeleteBuffers(1, &room_buffer_);
    glDeleteTextures(2, room_textures_.data());
    glDeleteFramebuffers(1, &frame_buffer_);
    glDeleteTextures(1, &color_buffer_);
    glDeleteRenderbuffers(1, &depth_buffer_);
  }
  eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
  if (surface_ != EGL_NO_SURFACE) eglDestroySurface(display_, surface_);
  eglTerminate(display_);
}

auto FieldRenderer::Render(const Eigen::Matrix4d& world_to_camera, bool fuel_labels,
                          const std::vector<std::array<double, 3>>* probes) -> cv::Mat {
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
  if (fuel_labels) glClearColor(0, 0, 0, 1);
  else glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glUseProgram(program_);
  glUniform1i(glGetUniformLocation(program_, "fuel_labels"), fuel_labels);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_buffer_);
  const auto camera = world_to_camera.cast<float>();
  const GLint mvp_uniform = glGetUniformLocation(program_, "mvp");
  const GLint model_uniform = glGetUniformLocation(program_, "model");
  const GLint color_uniform = glGetUniformLocation(program_, "color");
  const GLint soot_uniform = glGetUniformLocation(program_, "soot");
  const GLint seed_uniform = glGetUniformLocation(program_, "appearance_seed");
  const GLint material_uniform = glGetUniformLocation(program_, "material");
  glUniform1i(glGetUniformLocation(program_, "room_texture"), 0);
  auto draw_one = [&](const Draw& draw, std::size_t label = 0, GLenum mode = GL_TRIANGLES) {
    glBindBuffer(GL_ARRAY_BUFFER, draw.buffer ? draw.buffer : vertex_buffer_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, draw.buffer ? draw.buffer : vertex_buffer_);
    glUniform1i(material_uniform, draw.material);
    if (draw.material) glBindTexture(GL_TEXTURE_2D, room_textures_[draw.material - 1]);
    const Matrix mvp = projection_ * camera * draw.model;
    glUniformMatrix4fv(mvp_uniform, 1, GL_FALSE, mvp.data());
    glUniformMatrix4fv(model_uniform, 1, GL_FALSE, draw.model.data());
    const cv::Vec4f label_color{static_cast<float>(label & 255) / 255,
        static_cast<float>((label >> 8) & 255) / 255,
        static_cast<float>((label >> 16) & 255) / 255, 1};
    glUniform4fv(color_uniform, 1, fuel_labels ? label_color.val : draw.color.val);
    glUniform1f(soot_uniform, label && !fuel_labels ? soot_ : 0.f);
    glUniform1f(seed_uniform, float(appearance_seed_ % 65536) + float(label) * 13.37f);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, draw.vertex_stride,
                          reinterpret_cast<const void*>(static_cast<uintptr_t>(draw.position_offset)));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, draw.vertex_stride,
                          reinterpret_cast<const void*>(static_cast<uintptr_t>(draw.normal_offset)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glDrawElements(mode, draw.index_count, draw.index_type,
                   reinterpret_cast<const void*>(draw.index_offset));
  };
  for (const auto& draw : draws_) {
    if (!draw.transparent) draw_one(draw);
  }
  if (probes && !staged_fuel_draws_.empty()) for (std::size_t i = 0; i < probes->size(); ++i) {
    // Collapsed Fuel vertices rasterize a one-pixel point, depth-tested only
    // against the field. Fuel itself must not hide its cluster's center.
    auto probe = staged_fuel_draws_.front();
    probe.model.block<3, 3>(0, 0) *= 1e-6f;
    for (int axis = 0; axis < 3; ++axis) probe.model(axis, 3) = probes->at(i)[axis];
    draw_one(probe, i + 1, GL_POINTS);
  }
  else if (!probes) for (std::size_t i = 0; i < active_fuel_draws_.size(); ++i) {
    if (!active_fuel_draws_[i].transparent) draw_one(active_fuel_draws_[i], i + 1);
  }
  std::vector<const Draw*> transparent;
  for (const auto& draw : draws_) {
    if (draw.transparent) transparent.push_back(&draw);
  }
  for (const auto& draw : active_fuel_draws_) {
    if (draw.transparent) transparent.push_back(&draw);
  }
  std::sort(transparent.begin(), transparent.end(), [&](const Draw* a, const Draw* b) {
    return (camera * a->model)(2, 3) > (camera * b->model)(2, 3);
  });
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(GL_FALSE);
  if (!fuel_labels) for (const auto* draw : transparent) draw_one(*draw);
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  if (const GLenum error = glGetError(); error != GL_NO_ERROR)
    throw std::runtime_error("OpenGL render error " + std::to_string(error));
  cv::Mat rgba(image_size_, CV_8UC4);
  glReadPixels(0, 0, image_size_.width, image_size_.height,
               GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
  cv::flip(rgba, rgba, 0);
  cv::Mat bgr;
  cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
  return bgr;
}

}  // namespace path_camera_sim
