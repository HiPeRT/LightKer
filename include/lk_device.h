/*
 *  LightKer - Light and flexible GPU persistent threads library
 *  Copyright (C) 2016  Paolo Burgio
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __LK_DEVICE_H__
#define __LK_DEVICE_H__

__device__ uint32_t lkGetClusterID();
__device__ uint32_t lkGetCoreID();

#endif /* __LK_DEVICE_H__ */
