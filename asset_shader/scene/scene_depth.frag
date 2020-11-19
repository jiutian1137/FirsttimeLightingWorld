#version 330 core
in float fdepth;
void main(){ gl_FragDepth = fdepth; }