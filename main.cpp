// Team:
//Aasiya Qadri 
//Uroosa Lakhani 
//Ashraqat Mansour 

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define GLEW_STATIC
#define PLATFORM_OSX 1

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "OBJloader.h"    

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

// Camera setup
glm::vec3 camPos(0.0f, 0.0f, 6.0f);
glm::vec3 camUp(0.0f, 1.0f, 0.0f);
float fov = 45.0f;
float cameraSpeed = 1.5f;
float cameraHorizontalAngle = 90.0f;
float cameraVerticalAngle = 0.0f;
int fbWidth, fbHeight;
bool ignoreFirstMouseDelta = false;


// Data types 
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoords;
};

struct ShootingStar {
    glm::vec3 position;
    glm::vec3 velocity;
    float life;
    std::vector<glm::vec3> trail;
};

std::vector<ShootingStar> stars;

// Projectiles & Crosshair 
struct Projectile {
    glm::vec3 position;
    glm::vec3 velocity;
    float life;            
};
std::vector<Projectile> projectiles;

// Visibility flags for removable objects
bool showEarth = true;
bool showMoon = true;
bool showAstronaut = true;


// Crosshair resources
GLuint crosshairVAOInner = 0, crosshairVBOInner = 0; GLuint crosshairVAOOuter = 0, crosshairVBOOuter = 0; GLuint crosshairVAOPlate = 0, crosshairVBOPlate = 0;
GLuint progCrosshair = 0;

// Mouse click edge detector
bool wasMouseDown = false;

// Sphere generator 
std::vector<Vertex> generateSphere(int sectorCount, int stackCount, float radius = 0.5f) {
    std::vector<Vertex> vertices;
    const float pi = glm::pi<float>();
    for (int i = 0; i < stackCount; ++i) {
        float stackAngle1 = pi / 2 - i * pi / stackCount;
        float stackAngle2 = pi / 2 - (i + 1) * pi / stackCount;
        float y1 = radius * sinf(stackAngle1);
        float y2 = radius * sinf(stackAngle2);
        float r1 = radius * cosf(stackAngle1);
        float r2 = radius * cosf(stackAngle2);
        float t1 = 1.0f - (float)i / stackCount;
        float t2 = 1.0f - (float)(i + 1) / stackCount;

        for (int j = 0; j <= sectorCount; ++j) {
            float sectorAngle = j * 2 * pi / sectorCount;
            float x1 = r1 * cosf(sectorAngle);
            float z1 = r1 * sinf(sectorAngle);
            float x2 = r2 * cosf(sectorAngle);
            float z2 = r2 * sinf(sectorAngle);
            float s  = (float)j / sectorCount;

            glm::vec3 p1(x1, y1, z1);
            glm::vec3 p2(x2, y2, z2);

            vertices.push_back({ p1, glm::normalize(p1), glm::vec2(s, t1) });
            vertices.push_back({ p2, glm::normalize(p2), glm::vec2(s, t2) });
        }
    }
    return vertices;
}

// Shader utils 
static GLuint compileProgram(const char* vs, const char* fs) {
    auto compile = [](GLenum type, const char* src){
        GLuint sh = glCreateShader(type);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if(!ok){ char log[2048]; glGetShaderInfoLog(sh, 2048, nullptr, log); std::cerr << (type==GL_VERTEX_SHADER?"VS":"FS") << " error:\n" << log << std::endl; }
        return sh;
    };
    GLuint v = compile(GL_VERTEX_SHADER, vs);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram(); glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){ char log[2048]; glGetProgramInfoLog(p, 2048, nullptr, log); std::cerr << "Link error:\n" << log << std::endl; }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// VAO/VBO for interleaved Vertex
static GLuint createVAO(const std::vector<Vertex>& vertices) {
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    return VAO;
}

// Texture loading 
static GLuint loadTexture2D(const char* path, bool srgb=false) {
    GLuint tex; glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int w,h,n; stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path, &w, &h, &n, 0);
    if(!data){ std::cerr << "Failed to load texture: " << path << std::endl; return 0; }
    GLenum fmt = n==4?GL_RGBA:GL_RGB;
    GLenum ifmt = (srgb? (n==4?GL_SRGB_ALPHA:GL_SRGB) : fmt);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(data);
    return tex;
}

// Model loading (OBJ)
static GLuint setupModelVBO(const std::string& path, int& vertexCount) {
    std::vector<glm::vec3> vtx; std::vector<glm::vec3> nrm; std::vector<glm::vec2> uv;
    loadOBJ(path.c_str(), vtx, nrm, uv);
    GLuint VAO; glGenVertexArrays(1, &VAO); glBindVertexArray(VAO);
    GLuint vbo[3]; glGenBuffers(3, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); glBufferData(GL_ARRAY_BUFFER, vtx.size()*sizeof(glm::vec3), vtx.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0); glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); glBufferData(GL_ARRAY_BUFFER, nrm.size()*sizeof(glm::vec3), nrm.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,(void*)0); glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]); glBufferData(GL_ARRAY_BUFFER, uv.size()*sizeof(glm::vec2), uv.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,0,(void*)0); glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    vertexCount = (int)vtx.size();
    return VAO;
}

// HDR/Bloom FBO helpers
struct FBO {
    GLuint fbo=0, color=0, rboDepth=0;
    int w=0, h=0;
};

static FBO createHDRFBO(int w, int h) {
    FBO out; out.w=w; out.h=h;
    glGenFramebuffers(1, &out.fbo); glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
    glGenTextures(1, &out.color); glBindTexture(GL_TEXTURE_2D, out.color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out.color, 0);
    glGenRenderbuffers(1, &out.rboDepth); glBindRenderbuffer(GL_RENDERBUFFER, out.rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, out.rboDepth);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE) std::cerr<<"HDR FBO incomplete\n";
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return out;
}

struct PingPong {
    GLuint fbo[2]{0,0};
    GLuint tex[2]{0,0};
    int w=0,h=0;
};

static PingPong createPingPong(int w, int h) {
    PingPong p; p.w=w; p.h=h;
    glGenFramebuffers(2, p.fbo); glGenTextures(2, p.tex);
    for(int i=0;i<2;++i){
        glBindTexture(GL_TEXTURE_2D, p.tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindFramebuffer(GL_FRAMEBUFFER, p.fbo[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, p.tex[i], 0);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE) std::cerr<<"PingPong FBO incomplete\n";
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return p;
}

// Shaders 
static const char* VS_LIT = R"GLSL(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTex;
uniform mat4 model, view, projection;
out vec3 vFragPos;
out vec3 vNormal;
out vec2 vUV;
void main(){
    vec4 wpos = model * vec4(aPos,1.0);
    vFragPos = wpos.xyz;
    vNormal  = mat3(transpose(inverse(model))) * aNormal;
    vUV = aTex;
    gl_Position = projection * view * wpos;
    gl_PointSize = 12.0;
}
)GLSL";

static const char* FS_LIT = R"GLSL(
#version 330 core
in vec3 vFragPos; in vec3 vNormal; in vec2 vUV;
out vec4 FragColor;

uniform sampler2D texture1; // earth/moon
uniform sampler2D noise1;   // noise for Sun
uniform sampler2D noise2;   // noise for Sun

uniform bool useTexture;
uniform bool emissive;          // treat as Sun when true
uniform vec3  albedo;           // base color when not textured
uniform vec3  lightPos;         // Sun world pos
uniform vec3  viewPos;          // camera pos
uniform float shininess;        // spec power
uniform float time;

void main(){
    vec3 base = useTexture ? texture(texture1, vUV).rgb : albedo;

    if (emissive) {
        // Sun shading: limb darkening + animated granulation
        vec3 N = normalize(vNormal);
        vec3 V = normalize(viewPos - vFragPos);
        float mu = max(dot(N,V), 0.0);
        float a = 0.55; // center power
        float b = 0.55; // rim dim
        float limb = mix(b, 1.0, pow(mu, a));

        // Build a tangent-ish UV from world pos
        vec3 T = normalize(vec3(1.0,0.0,0.0));
        vec3 B = normalize(cross(N, T));
        vec2 suv = vec2(dot(vFragPos, T), dot(vFragPos, B));
        vec2 uv1 = suv * 2.0 + vec2(time*0.03, 0.0);
        vec2 uv2 = suv * 3.5 + vec2(0.0, -time*0.02);
        float n1 = texture(noise1, uv1).r;
        float n2 = texture(noise2, uv2 + 0.15*n1).r;
        float granulation = smoothstep(0.35, 0.75, n1*0.6 + n2*0.5);
        float sunspots = smoothstep(0.45, 0.4, n1*n2);

        vec3 hotWhite = vec3(1.0, 0.98, 0.92);
        vec3 warmEdge = vec3(1.0, 0.82, 0.45);
        vec3 colorRamp = mix(warmEdge, hotWhite, pow(mu, 0.7));
        vec3 detail = mix(vec3(0.8), vec3(1.1), granulation) - 0.25*sunspots;
        vec3 result = colorRamp * limb * detail * 8.0; // bright, for bloom
        FragColor = vec4(result, 1.0);
        return;
    }

    // Regular Blinn-Phong for planets/astronaut
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightPos - vFragPos);
    float diff = max(dot(N,L), 0.0);
    vec3 V = normalize(viewPos - vFragPos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N,H), 0.0), shininess);
    vec3 ambient  = 0.05 * base;
    vec3 diffuse  = diff * base;
    vec3 specular = spec * vec3(0.6);
    FragColor = vec4(ambient + diffuse + specular, 1.0);
}
)GLSL";

// Fullscreen/background quad
static const char* VS_BG = R"GLSL(
#version 330 core
layout (location = 0) in vec3 aPos; layout (location = 1) in vec2 aTex;
out vec2 vUV; void main(){ vUV=aTex; gl_Position=vec4(aPos,1.0); }
)GLSL";
static const char* FS_BG = R"GLSL(
#version 330 core
in vec2 vUV; out vec4 FragColor; uniform sampler2D backgroundTex;
void main(){ FragColor = texture(backgroundTex, vUV); }
)GLSL";

// Bright pass (extract highlights)
static const char* VS_QUAD = R"GLSL(
#version 330 core
layout (location=0) in vec3 aPos; layout (location=1) in vec2 aTex; out vec2 vUV;
void main(){ vUV=aTex; gl_Position=vec4(aPos,1.0); }
)GLSL";
static const char* FS_BRIGHT = R"GLSL(
#version 330 core
in vec2 vUV; out vec4 FragColor; uniform sampler2D hdrColor; uniform float threshold;
void main(){ vec3 c = texture(hdrColor, vUV).rgb; vec3 bright = max(c - vec3(threshold), vec3(0.0)); FragColor=vec4(bright,1.0); }
)GLSL";

// Gaussian blur (separable)
static const char* FS_BLUR = R"GLSL(
#version 330 core
in vec2 vUV; out vec4 FragColor; uniform sampler2D image; uniform bool horizontal; uniform float texelW; uniform float texelH;
void main(){
    float w[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec2 texel = horizontal ? vec2(texelW,0.0) : vec2(0.0,texelH);
    vec3 acc = texture(image, vUV).rgb * w[0];
    for(int i=1;i<5;++i){ acc += texture(image, vUV + texel*float(i)).rgb * w[i]; acc += texture(image, vUV - texel*float(i)).rgb * w[i]; }
    FragColor = vec4(acc,1.0);
}
)GLSL";

// Composite + tone-map (ACES-ish)
static const char* FS_COMPOSITE = R"GLSL(
#version 330 core
in vec2 vUV; out vec4 FragColor;
uniform sampler2D sceneTex; uniform sampler2D bloomTex; uniform float exposure;
vec3 ACES(vec3 x){
    const float A=2.51, B=0.03, C=2.43, D=0.59, E=0.14; return clamp((x*(A*x+B))/(x*(C*x+D)+E), 0.0, 1.0);
}
void main(){
    vec3 hdr = texture(sceneTex, vUV).rgb + texture(bloomTex, vUV).rgb;
    vec3 mapped = ACES(hdr * exposure);
    mapped = pow(mapped, vec3(1.0/2.2)); // gamma
    FragColor = vec4(mapped, 1.0);
}
)GLSL";

// Corona billboard (additive sprite around the sun)
static const char* VS_CORONA = R"GLSL(
#version 330 core
layout (location=0) in vec3 aPos; layout (location=1) in vec2 aTex;
uniform mat4 model; uniform mat4 view; uniform mat4 projection; out vec2 vUV;
void main(){ vUV=aTex; gl_Position = projection * view * model * vec4(aPos,1.0); }
)GLSL";
static const char* FS_CORONA = R"GLSL(
#version 330 core
in vec2 vUV; out vec4 FragColor;
uniform sampler2D noise1; uniform float time;
void main(){
    // radial coords in -1..1
    vec2 p = vUV*2.0 - 1.0; float r = length(p);
    float radial = pow(smoothstep(1.0, 0.0, r), 2.0);
    float rings = texture(noise1, p*1.5 + vec2(time*0.05)).r;
    float filaments = smoothstep(0.3, 0.9, rings);
    vec3 inner = vec3(1.0, 0.95, 0.9);
    vec3 outer = vec3(1.0, 0.70, 0.30);
    vec3 col = mix(outer, inner, smoothstep(0.0, 0.7, 1.0 - r));
    vec3 glow = col * radial * (0.6 + 0.7*filaments);
    FragColor = vec4(glow, 1.0); // use additive blend in scene pass
}
)GLSL";

// Main 

// ---- Crosshair shaders (NDC pass-through) ----
static const char* VS_CROSSHAIR = R"GLSL(
#version 330 core
layout (location=0) in vec2 aPos; // Already in NDC
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
)GLSL";

static const char* FS_CROSSHAIR = R"GLSL(
#version 330 core
out vec4 FragColor;
uniform vec4 uColor;
void main() { FragColor = uColor; }
)GLSL";

int main(){
    srand(static_cast<unsigned>(time(0)));
    if(!glfwInit()){ std::cerr<<"GLFW init failed\n"; return -1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Sun with Bloom", nullptr, nullptr);
    if(!window){ std::cerr<<"Window creation failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    int initW, initH; glfwGetWindowSize(window, &initW, &initH); glfwSetCursorPos(window, initW*0.5, initH*0.5);
    ignoreFirstMouseDelta = true; 
    glewExperimental = GL_TRUE; if(glewInit()!=GLEW_OK){ std::cerr<<"GLEW init failed\n"; return -1; }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Shaders
    GLuint progLit = compileProgram(VS_LIT, FS_LIT);
    GLuint progBG  = compileProgram(VS_BG,  FS_BG);
    GLuint progBright = compileProgram(VS_QUAD, FS_BRIGHT);
    GLuint progBlur   = compileProgram(VS_QUAD, FS_BLUR);
    GLuint progComposite = compileProgram(VS_QUAD, FS_COMPOSITE);
    GLuint progCorona = compileProgram(VS_CORONA, FS_CORONA);

    // Crosshair program and VAOs  
    progCrosshair = compileProgram(VS_CROSSHAIR, FS_CROSSHAIR);
    {
        auto buildCrosshairVAO = [](float s, float g, float t, GLuint &vao, GLuint &vbo){
            float verts[] = {
                // LEFT arm
                -g - s, -t,   -g, -t,   -g,  t,
                -g - s, -t,   -g,  t,   -g - s,  t,
                // RIGHT arm
                 g, -t,    g + s, -t,    g + s,  t,
                 g, -t,    g + s,  t,     g,      t,
                // DOWN arm
                -t, -g - s,   t, -g - s,   t, -g,
                -t, -g - s,   t, -g,      -t, -g,
                // UP arm
                -t,  g,       t,  g,       t,  g + s,
                -t,  g,       t,  g + s,  -t,  g + s
            };
            glGenVertexArrays(1, &vao);
            glGenBuffers(1, &vbo);
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glBindVertexArray(0);
        };

        // Base geometry dims in NDC
        float s_in = 0.06f;    // arm length
        float g_in = 0.010f;   // center gap half-size
        float t_in = 0.010f;   // half thickness

        // Outline amounts
        float outline_t = 0.006f; // extra thickness
        float outline_s = 0.012f; // extra length

        // inner red & outer black VAOs
        buildCrosshairVAO(s_in + outline_s, g_in, t_in + outline_t, crosshairVAOOuter, crosshairVBOOuter);
        buildCrosshairVAO(s_in,             g_in, t_in, crosshairVAOInner, crosshairVBOInner);

        // Center opaque, black square for visibility on bright backgrounds
        {
            float p = 0.018f; // halfsize of the square in NDC
            float plate[] = {
                -p, -p,   p, -p,   p,  p,
                -p, -p,   p,  p,  -p,  p
            };
            glGenVertexArrays(1, &crosshairVAOPlate);
            glGenBuffers(1, &crosshairVBOPlate);
            glBindVertexArray(crosshairVAOPlate);
            glBindBuffer(GL_ARRAY_BUFFER, crosshairVBOPlate);
            glBufferData(GL_ARRAY_BUFFER, sizeof(plate), plate, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glBindVertexArray(0);
        }
    
    }
    
    
    // Geometry
    std::vector<Vertex> sphere = generateSphere(72, 36);
    GLuint sphereVAO = createVAO(sphere);
    int sphereVertexCount = (int)sphere.size();

    // Fullscreen / generic quad
    float quadVertices[] = {
        // pos                // uv
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f,
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f
    };
    GLuint quadVAO, quadVBO; glGenVertexArrays(1,&quadVAO); glGenBuffers(1,&quadVBO);
    glBindVertexArray(quadVAO); glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    // Corona quad (unit quad centered) — reuse quadVAO but with a model matrix that scales/centers

    // Textures
    GLuint texEarth = loadTexture2D("earth.jpg", true);
    GLuint texMoon  = loadTexture2D("moon.jpg",  true);
    GLuint texBG    = loadTexture2D("bg.jpg",    true);
    GLuint texN1    = loadTexture2D("noise1.png", false);
    GLuint texN2    = loadTexture2D("noise2.png", false);

    // Astronaut model
    int astroCount=0; GLuint astroVAO = setupModelVBO("Models/astronaut.obj", astroCount);

    // Uniform locations for lit
    glm::mat4 projection = glm::perspective(glm::radians(fov), (float)fbWidth / (float)fbHeight, 0.1f, 100.0f);

    // HDR/Bloom FBOs
    FBO hdr = createHDRFBO(fbWidth, fbHeight);
    PingPong pp = createPingPong(fbWidth, fbHeight);

    // Static uniforms
    const glm::vec3 sunWorldPos(0.0f);
    {
        glm::vec3 dir = glm::normalize(sunWorldPos - camPos); // from camera to Sun
        cameraVerticalAngle   = glm::degrees(asinf(dir.y));             // pitch
        cameraHorizontalAngle = glm::degrees(atan2f(-dir.z, dir.x));    // yaw (note the -z to match your direction formula)
    }
    // Timing
    // removed initial cursor position
    float lastTime = (float)glfwGetTime();
    
    while(!glfwWindowShouldClose(window)){
        float timeNow = (float)glfwGetTime(); float dt = timeNow - lastTime; lastTime=timeNow;


        // Camera controls (center-locked cursor) 
        int winW, winH; glfwGetWindowSize(window, &winW, &winH);
        double centerX = winW * 0.5, centerY = winH * 0.5;

        double mx, my; glfwGetCursorPos(window, &mx, &my);
        double dx = mx - centerX;
        double dy = my - centerY;

        if (ignoreFirstMouseDelta) { 
            dx = 0.0; dy = 0.0;
            ignoreFirstMouseDelta = false;
        }

        float cameraAngularSpeed = 0.12f; // for center-based delta
        cameraHorizontalAngle -= (float)dx * cameraAngularSpeed;
        cameraVerticalAngle   += (float)dy * cameraAngularSpeed; // FPS inverted-Y

        // Clamp angles
        cameraVerticalAngle = std::max(-89.0f, std::min(89.0f, cameraVerticalAngle));
        cameraHorizontalAngle = std::max(60.0f, std::min(120.0f, cameraHorizontalAngle));

        // Recenter cursor so deltas are relative to screen center
        glfwSetCursorPos(window, centerX, centerY);

        // Build camera basis
        glm::vec3 direction(
            cos(glm::radians(cameraVerticalAngle)) * cos(glm::radians(cameraHorizontalAngle)),
            sin(glm::radians(cameraVerticalAngle)),
            -cos(glm::radians(cameraVerticalAngle)) * sin(glm::radians(cameraHorizontalAngle))
        );
        glm::vec3 right = glm::normalize(glm::cross(direction, camUp));
        glm::vec3 up    = glm::normalize(glm::cross(right, direction));

        // Keyboard movement
        if(glfwGetKey(window, GLFW_KEY_W)==GLFW_PRESS) camPos += direction*dt*cameraSpeed;
        if(glfwGetKey(window, GLFW_KEY_S)==GLFW_PRESS) camPos -= direction*dt*cameraSpeed;
        if(glfwGetKey(window, GLFW_KEY_D)==GLFW_PRESS) camPos += right*dt*cameraSpeed;
        if(glfwGetKey(window, GLFW_KEY_A)==GLFW_PRESS) camPos -= right*dt*cameraSpeed;
        if(glfwGetKey(window, GLFW_KEY_E)==GLFW_PRESS) camPos += up*dt*cameraSpeed;
        if(glfwGetKey(window, GLFW_KEY_Q)==GLFW_PRESS) camPos -= up*dt*cameraSpeed;

        glm::mat4 view = glm::lookAt(camPos, camPos+direction, up);


        // Shoot projectiles from crosshair with left click 
        {
            int mouseState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
            bool isDown = (mouseState == GLFW_PRESS);
            if (isDown && !wasMouseDown) {
                Projectile p;
                const float spawnOffset = 0.2f;
                const float projSpeed   = 12.0f;
                // direction computed above
                p.position = camPos + glm::normalize(direction) * spawnOffset;
                p.velocity = glm::normalize(direction) * projSpeed;
                p.life     = 3.0f;
                projectiles.push_back(p);
            }
            wasMouseDown = isDown;
        }


        // Scene pass into HDR FBO 
        glBindFramebuffer(GL_FRAMEBUFFER, hdr.fbo);
        
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);
        //glViewport(0,0,W,H);
        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Background (drawn first without depth)
        glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE);
        glUseProgram(progBG);
        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texBG);
        glUniform1i(glGetUniformLocation(progBG,"backgroundTex"), 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        glEnable(GL_CULL_FACE); glEnable(GL_DEPTH_TEST);

        // Common lit uniforms
        glUseProgram(progLit);
        glUniformMatrix4fv(glGetUniformLocation(progLit,"view"),1,GL_FALSE,glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(progLit,"projection"),1,GL_FALSE,glm::value_ptr(projection));
        glUniform3fv(glGetUniformLocation(progLit,"lightPos"),1,glm::value_ptr(sunWorldPos));
        glUniform3fv(glGetUniformLocation(progLit,"viewPos"),1,glm::value_ptr(camPos));
        glUniform1f(glGetUniformLocation(progLit,"shininess"), 64.0f);
        glUniform1f(glGetUniformLocation(progLit,"time"), timeNow);
        glUniform1i(glGetUniformLocation(progLit,"texture1"),0);
        glUniform1i(glGetUniformLocation(progLit,"noise1"),1);
        glUniform1i(glGetUniformLocation(progLit,"noise2"),2);

        // Sun (emissive)
        glBindVertexArray(sphereVAO);
        glm::mat4 sunModel = glm::scale(glm::mat4(1.0f), glm::vec3(1.5f));
        glUniformMatrix4fv(glGetUniformLocation(progLit,"model"),1,GL_FALSE,glm::value_ptr(sunModel));
        glUniform1i(glGetUniformLocation(progLit,"useTexture"), GL_FALSE);
        glUniform1i(glGetUniformLocation(progLit,"emissive"), GL_TRUE);
        glUniform3f(glGetUniformLocation(progLit,"albedo"), 1.0f, 1.0f, 0.0f);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texN1);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, texN2);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, sphereVertexCount);
        glActiveTexture(GL_TEXTURE0); // reset to unit 0 for planet textures

        if(showEarth){
        // Earth (textured, lit)
        glm::mat4 earthModel = glm::rotate(glm::mat4(1.0f), timeNow * glm::radians(30.0f), glm::vec3(0,1,0));
        earthModel = glm::translate(earthModel, glm::vec3(2.3f, 0.0f, 0.0f));
        earthModel = glm::rotate(earthModel, timeNow * glm::radians(100.0f), glm::vec3(0,1,0));
        glUniformMatrix4fv(glGetUniformLocation(progLit,"model"),1,GL_FALSE,glm::value_ptr(earthModel));
        glUniform1i(glGetUniformLocation(progLit,"useTexture"), GL_TRUE);
        glUniform1i(glGetUniformLocation(progLit,"emissive"), GL_FALSE);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texEarth);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, sphereVertexCount);
        }

        if(showMoon){
        // Moon (textured, lit)
        glm::mat4 moonModel = glm::rotate(glm::mat4(1.0f), timeNow * glm::radians(30.0f), glm::vec3(0,1,0));
        moonModel = glm::translate(moonModel, glm::vec3(2.3f, 0.0f, 0.0f));
        moonModel = glm::rotate(moonModel, timeNow * glm::radians(100.0f), glm::vec3(0,1,0));
        moonModel = glm::translate(moonModel, glm::vec3(1.0f, 0.0f, 0.0f));
        moonModel = glm::scale(moonModel, glm::vec3(0.6f));
        glUniformMatrix4fv(glGetUniformLocation(progLit,"model"),1,GL_FALSE,glm::value_ptr(moonModel));
        glUniform1i(glGetUniformLocation(progLit,"useTexture"), GL_TRUE);
        glUniform1i(glGetUniformLocation(progLit,"emissive"), GL_FALSE);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texMoon);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, sphereVertexCount);
        }

        if(showAstronaut){
        // Astronaut (untextured, lit)
        glm::mat4 astroModel = glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, -1.0f, -4.0f));
        astroModel = glm::scale(astroModel, glm::vec3(0.02f));
        glm::vec3 astroPos = glm::vec3(5.0f, -1.0f, -4.0f);
        glm::vec3 toCamera = glm::normalize(camPos - astroPos);
        float angle = atan2(toCamera.x, -toCamera.z) + glm::radians(120.0f);
        astroModel = glm::rotate(astroModel, angle, glm::vec3(0.0f,1.0f,0.0f));
        astroModel = glm::scale(astroModel, glm::vec3(0.5f));
        glUniformMatrix4fv(glGetUniformLocation(progLit,"model"),1,GL_FALSE,glm::value_ptr(astroModel));
        glUniform1i(glGetUniformLocation(progLit,"useTexture"), GL_FALSE);
        glUniform1i(glGetUniformLocation(progLit,"emissive"), GL_FALSE);
        glUniform3f(glGetUniformLocation(progLit,"albedo"), 0.7f, 0.7f, 0.9f);
        glBindVertexArray(astroVAO);
        glDrawArrays(GL_TRIANGLES, 0, astroCount);
        }

        // Shooting stars (points)
        glBindVertexArray(sphereVAO); // to keep attrib 0 bound
        static float spawnAccum=0.0f; spawnAccum+=dt;
        if(spawnAccum>1.0f){ spawnAccum=0.0f; ShootingStar s; s.position=glm::vec3(-5.0f+rand()%10, 3.0f+rand()%2, -5.0f); s.velocity=glm::vec3(2.5f,-3.0f,0.0f); s.life=2.0f; stars.push_back(s);}        
        for(auto& s: stars){ s.trail.push_back(s.position); if(s.trail.size()>10) s.trail.erase(s.trail.begin()); s.position += s.velocity*dt; s.life-=dt; }
        stars.erase(std::remove_if(stars.begin(), stars.end(), [](const ShootingStar& s){ return s.life<=0.0f; }), stars.end());

        glUniform1i(glGetUniformLocation(progLit,"useTexture"), GL_FALSE);
        glUniform1i(glGetUniformLocation(progLit,"emissive"), GL_TRUE);
        for(const auto& s: stars){
            float alphaStep = 1.0f / std::max<size_t>(1, s.trail.size());
            float alpha = 0.0f;
            for(const auto& p: s.trail){
                glm::mat4 m = glm::translate(glm::mat4(1.0f), p);
                glUniformMatrix4fv(glGetUniformLocation(progLit,"model"),1,GL_FALSE,glm::value_ptr(m));
                glUniform3f(glGetUniformLocation(progLit,"albedo"), alpha, alpha, alpha);
                glDrawArrays(GL_POINTS, 0, 1);
                alpha += alphaStep;
            }
            glm::mat4 mh = glm::translate(glm::mat4(1.0f), s.position);
            glUniformMatrix4fv(glGetUniformLocation(progLit,"model"),1,GL_FALSE,glm::value_ptr(mh));
            glUniform3f(glGetUniformLocation(progLit,"albedo"), 1.5f, 1.5f, 1.5f);
            glDrawArrays(GL_POINTS, 0, 1);
        }


        // Projectiles update (with collision) 
        // Animated centers (match draw transforms)
        glm::mat4 earthModel_c = glm::rotate(glm::mat4(1.0f), timeNow * glm::radians(30.0f), glm::vec3(0,1,0));
        earthModel_c = glm::translate(earthModel_c, glm::vec3(2.3f, 0.0f, 0.0f));
        earthModel_c = glm::rotate(earthModel_c, timeNow * glm::radians(100.0f), glm::vec3(0,1,0));
        glm::vec3 earthCenter_c = glm::vec3(earthModel_c * glm::vec4(0,0,0,1));
        float earthRadius_c = 0.60f;

        glm::mat4 moonModel_c = glm::rotate(glm::mat4(1.0f), timeNow * glm::radians(30.0f), glm::vec3(0,1,0));
        moonModel_c = glm::translate(moonModel_c, glm::vec3(2.3f, 0.0f, 0.0f));
        moonModel_c = glm::rotate(moonModel_c, timeNow * glm::radians(100.0f), glm::vec3(0,1,0));
        moonModel_c = glm::translate(moonModel_c, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec3 moonCenter_c = glm::vec3(moonModel_c * glm::vec4(0,0,0,1));
        float moonRadius_c = 0.10f;

        glm::vec3 astroCenter_c = glm::vec3(5.0f, -1.0f, -4.0f);
        float astroRadius_c = 0.80f;
        float hitSlack = 0.50f;

        for (auto& b : projectiles) {
            b.position += b.velocity * dt;
            b.life -= dt;
            if (showEarth && glm::length(b.position - earthCenter_c) <= earthRadius_c + hitSlack) { showEarth = false; b.life = 0.0f; }
            if (showMoon && glm::length(b.position - moonCenter_c) <= moonRadius_c + hitSlack)   { showMoon = false;  b.life = 0.0f; }
            if (showAstronaut && glm::length(b.position - astroCenter_c) <= astroRadius_c + hitSlack) { showAstronaut = false; b.life = 0.0f; }
        }
        projectiles.erase(
            std::remove_if(projectiles.begin(), projectiles.end(),
                           [](const Projectile& b){ return b.life <= 0.0f; }),
            projectiles.end()
        );
        // Projectiles render 
        glUseProgram(progLit);
        glUniform1i(glGetUniformLocation(progLit,"useTexture"), GL_FALSE);
        glUniform1i(glGetUniformLocation(progLit,"emissive"),   GL_TRUE);
        glBindVertexArray(sphereVAO); // ensure a VAO is bound
        for (const auto& b : projectiles) {
            glm::mat4 m = glm::translate(glm::mat4(1.0f), b.position);
            glUniformMatrix4fv(glGetUniformLocation(progLit,"model"),1,GL_FALSE,glm::value_ptr(m));
            glUniform3f(glGetUniformLocation(progLit,"albedo"), 2.5f, 0.0f, 0.0f); 
            glDrawArrays(GL_POINTS, 0, 1);
        }
        glBindVertexArray(0);

        // Corona billboard 
        glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE);
        glUseProgram(progCorona);
        glUniform1i(glGetUniformLocation(progCorona,"noise1"), 0);
        glUniform1f(glGetUniformLocation(progCorona,"time"), timeNow);
        // Billboard model: center at sun, face camera using right/up
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, sunWorldPos);
        // overwrite basis with camera right/up/forward to face cam
        model[0] = glm::vec4(right*3.0f, 0.0f); // scale controls corona size
        model[1] = glm::vec4(up*3.0f,    0.0f);
        model[2] = glm::vec4(-direction, 0.0f);
        glUniformMatrix4fv(glGetUniformLocation(progCorona,"model"),1,GL_FALSE,glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(progCorona,"view"),1,GL_FALSE,glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(progCorona,"projection"),1,GL_FALSE,glm::value_ptr(projection));
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texN1);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        glDisable(GL_BLEND);

        // Bright pass 
        glBindFramebuffer(GL_FRAMEBUFFER, pp.fbo[0]);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(progBright);
        glUniform1i(glGetUniformLocation(progBright,"hdrColor"),0);
        glUniform1f(glGetUniformLocation(progBright,"threshold"), 1.0f);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, hdr.color);
        glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES,0,6);



        // Crosshair overlay (draw last on default framebuffer) 
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        // Disable MSAA to avoid soft edges (some drivers smooth triangle edges)
        glDisable(GL_MULTISAMPLE);

        glUseProgram(progCrosshair);

        // Opaque black at center
        glBindVertexArray(crosshairVAOPlate);
        glUniform4f(glGetUniformLocation(progCrosshair,"uColor"), 0.0f, 0.0f, 0.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Outer black outline quads
        glBindVertexArray(crosshairVAOOuter);
        glUniform4f(glGetUniformLocation(progCrosshair,"uColor"), 0.0f, 0.0f, 0.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, 0, 24);

        // Inner red quads
        glBindVertexArray(/* old_crosshairVAO */crosshairVAOInner);
        glUniform4f(glGetUniformLocation(progCrosshair,"uColor"), 1.0f, 0.0f, 0.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, 0, 24);

        glBindVertexArray(0);

        // Restore for future frames, if any UI later
        glEnable(GL_MULTISAMPLE);
        glEnable(GL_DEPTH_TEST);
    

        // Blur ping-pong 
        bool horizontal=true; int passes=8;
        glUseProgram(progBlur);
        for(int i=0;i<passes;++i){
            glBindFramebuffer(GL_FRAMEBUFFER, pp.fbo[horizontal?1:0]);
            glUniform1i(glGetUniformLocation(progBlur,"image"), 0);
            glUniform1i(glGetUniformLocation(progBlur,"horizontal"), horizontal);
            glUniform1f(glGetUniformLocation(progBlur,"texelW"), 1.0f/fbWidth);
            glUniform1f(glGetUniformLocation(progBlur,"texelH"), 1.0f/fbHeight);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, pp.tex[horizontal?0:1]);
        glDrawArrays(GL_TRIANGLES, 0, 24);
            horizontal = !horizontal;
        }

        // Composite to default framebuffer 
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fbWidth, fbHeight);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(progComposite);
        glUniform1i(glGetUniformLocation(progComposite,"sceneTex"), 0);
        glUniform1i(glGetUniformLocation(progComposite,"bloomTex"), 1);
        glUniform1f(glGetUniformLocation(progComposite,"exposure"), 0.7f);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, hdr.color);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, pp.tex[horizontal?0:1]);
        glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES,0,6);


        glfwSwapBuffers(window);
        glfwPollEvents();
        if(glfwGetKey(window, GLFW_KEY_ESCAPE)==GLFW_PRESS) glfwSetWindowShouldClose(window,true);
    }

    glfwTerminate();
    return 0;
}
