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

#ifndef __DATA_H__
#define __DATA_H__

#ifndef L_MAX_LENGTH
#define L_MAX_LENGTH 20
#endif /* L_MAX_LENGTH */


#ifndef WORK_TIME
#define WORK_TIME 200000
#endif

struct data_t
{
  char str[L_MAX_LENGTH];
};

struct res_t
{
  unsigned int num;
};

#endif /* __DATA_H__*/
