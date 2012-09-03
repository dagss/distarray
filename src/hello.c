#include <string.h>

#include "hello.h"

void say_hello(char *output, size_t maxlen) {
  strncpy(output, "Hello World!", maxlen);
}
