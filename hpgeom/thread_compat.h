// thread_compat.h
#ifndef THREAD_COMPAT_H
#define THREAD_COMPAT_H

#ifdef _WIN32
    #include <windows.h>
    #include <process.h>

    typedef HANDLE thread_handle_t;
    typedef unsigned (__stdcall *thread_func_t)(void *);

    #define THREAD_CALL __stdcall

    static inline int thread_create(thread_handle_t *thread, void *(*func)(void *), void *arg) {
        // Wrapper to convert calling convention
        struct wrapper_data {
            void *(*func)(void *);
            void *arg;
        };
        static unsigned __stdcall wrapper(void *data) {
            struct wrapper_data *wd = (struct wrapper_data *)data;
            void *(*f)(void *) = wd->func;
            void *a = wd->arg;
            free(wd);
            f(a);
            return 0;
        }
        struct wrapper_data *wd = malloc(sizeof(*wd));
        if (!wd) return -1;
        wd->func = func;
        wd->arg = arg;
        *thread = (HANDLE)_beginthreadex(NULL, 0, wrapper, wd, 0, NULL);
        return (*thread == 0) ? -1 : 0;
    }

    static inline int thread_join(thread_handle_t thread) {
        WaitForSingleObject(thread, INFINITE);
        CloseHandle(thread);
        return 0;
    }

#else
    #include <pthread.h>

    typedef pthread_t thread_handle_t;

    #define THREAD_CALL

    static inline int thread_create(thread_handle_t *thread, void *(*func)(void *), void *arg) {
        return pthread_create(thread, NULL, func, arg);
    }

    static inline int thread_join(thread_handle_t thread) {
        return pthread_join(thread, NULL);
    }

#endif

#endif // THREAD_COMPAT_H
