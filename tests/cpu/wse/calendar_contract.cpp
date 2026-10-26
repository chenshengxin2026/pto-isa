/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cassert>
#include <pto/wse/calendar_contract.hpp>

int main()
{
    using namespace pto::wse;
    using Plan = AllGatherPlan<8, 96, 1, 2, 5, 9, 12>;
    static_assert(Plan::row(9) == 2 && Plan::offset(2) == 384);
    static_assert(Plan::kRecvBytes == 6144 && Plan::kDstStride == 768);
    using One = AllGatherPlan<2, 16, 0, 7>;
    static_assert((One::kMembers - 1) * One::kSendBytes == 0);
    ReceiveContract recv{6144, 4608, 44};
    assert(recv.check(43, 1, 192, 8, 192, 768) == Error::Epoch);
    assert(recv.check(44, 1, 6144, 1, 192, 0) == Error::Capacity);
    assert(recv.check(44, 1, 192, 8, 192, 32) == Error::Geometry);
    assert(recv.check(44, 1, 192, 8, 192, 768) == Error::OK);
    recv.commit(1, 1536);
    assert(recv.check(44, 1, 192, 8, 192, 768) == Error::Duplicate);
    assert(recv.committed == 1536);
    assert(recv.check(44, 2, 384, 8, 192, 768) == Error::OK);
    recv.commit(2, 1536);
    recv.commit(3, 1536);
    assert(recv.expected == recv.committed);
    assert(recv.check(44, 0, 0, 8, 192, 768) == Error::ByteCount);
}
