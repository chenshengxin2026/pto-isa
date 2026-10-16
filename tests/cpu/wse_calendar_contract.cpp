// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <pto/wse/cpu.hpp>
#include <cassert>
#include <iostream>

template <class Fn>
void Reject(Fn fn)
{
    bool rejected = false;
    try {
        fn();
    } catch (const std::exception&) {
        rejected = true;
    }
    assert(rejected);
}

int main()
{
    using pto::wse::CpuReceiver;
    CpuReceiver receiver(128, 64, 1, 2);
    std::vector<uint8_t> data(32, 0xab);
    assert(receiver.Commit(0, 0, 0, data, 1, 2, false));
    assert(receiver.Committed() == 0 && !receiver.Complete());
    assert(receiver.Commit(1, 0, 32, data, 1, 2));
    assert(!receiver.Commit(1, 0, 32, data, 1, 2));
    assert(receiver.Committed() == 32 && !receiver.Complete());
    Reject([&] { receiver.Commit(2, 0, 64, data, 2, 2); });
    Reject([&] { receiver.Commit(2, 0, 64, data, 1, 1); });
    Reject([&] { receiver.Commit(2, 0, 120, data, 1, 2); });
    Reject([&] { receiver.Commit(2, 0, 32, data, 1, 2); });
    assert(receiver.Commit(2, 0, 64, data, 1, 2));
    assert(receiver.Complete() && receiver.Committed() == 64);
    Reject([&] { receiver.Commit(3, 0, 96, data, 1, 2); });
    Reject([&] { receiver.Commit(1, 0, 96, data, 1, 2); });
    Reject([&] { CpuReceiver invalid(64, 65, 1, 1); });
    Reject([&] { CpuReceiver invalid(64, 32, 256, 1); });
    std::cout << "PASS: route ABI, self copy, completion, duplicates, epoch, launch, bounds, overlap\n";
}
