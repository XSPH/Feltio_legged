#ifndef ENUMCLASS_H
#define ENUMCLASS_H

enum class UserCommand{
    NONE,
    FIXED,
    PASS,
    RL
};

enum class FSMMode{
    NORMAL,
    CHANGE
};

enum class FSMStateName{
    INVALID,
    PASSIVE,
    FIXEDSTAND,
    Rl
};

#endif

