#pragma once

#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>

class ThreadPool {
public:
    explicit ThreadPool(std::size_t count = 0)
        : _threadCount(count > 0 ? count : std::thread::hardware_concurrency())
    {
        _workers.reserve(_threadCount);
        for (std::size_t i = 0; i < _threadCount; i++) {
            _workers.emplace_back(&ThreadPool::workerLoop, this, i);
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(_mtx);
            _stop = true;
            ++_gen;
        }
        _startCv.notify_all();
        for (auto& w : _workers) w.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void runConcurrentTask(const std::function<void(std::size_t, std::size_t)>& func) {
        {
            std::unique_lock<std::mutex> lk(_mtx);
            _readyCv.wait(lk, [this] { return _activeWorkers == 0; });
            _task = func;
            _doneCount = 0;
            _activeWorkers = _threadCount;
            ++_gen;
        }
        _startCv.notify_all();

        {
            std::unique_lock<std::mutex> lk(_mtx);
            _doneCv.wait(lk, [this] { return _doneCount >= _threadCount; });
        }
    }

    std::size_t threadCount() const { return _threadCount; }

private:
    void workerLoop(std::size_t id) {
        std::size_t myGen = 0;
        while (true) {
            std::function<void(std::size_t, std::size_t)> task;
            {
                std::unique_lock<std::mutex> lk(_mtx);
                _startCv.wait(lk, [&] { return _gen > myGen || _stop; });
                if (_stop) return;
                myGen = _gen;
                task = _task;
            }

            task(id, _threadCount);

            {
                std::lock_guard<std::mutex> lk(_mtx);
                ++_doneCount;
                --_activeWorkers;
            }
            _doneCv.notify_one();
            _readyCv.notify_one();
        }
    }

    std::vector<std::thread> _workers;
    std::size_t _threadCount;
    bool _stop = false;
    std::size_t _gen = 0;
    std::size_t _doneCount = 0;
    std::size_t _activeWorkers = 0;
    std::function<void(std::size_t, std::size_t)> _task;
    std::mutex _mtx;
    std::condition_variable _startCv;
    std::condition_variable _doneCv;
    std::condition_variable _readyCv;
};
