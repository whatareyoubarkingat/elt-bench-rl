select t.id as train_id, t.direction,
  case when t.direction = 'east' and count(c.id) < 4 then 1 else 0 end as run_east_and_has_less_than_4_cars,
  case when sum(case when c.len = 'long' then 1 else 0 end) > 2 then 1 else 0 end as has_more_than_2_long_cars,
  max(case when c.position = 3 and c.sides = 'double' then 1 else 0 end) as has_double_sided_cars_in_3rd_position,
  max(case when c.position = 1 and c.shape = 'rectangle' then 1 else 0 end) as has_rectangular_shaped_head_cars,
  coalesce(sum(case when c.roof = 'none' then 1 else 0 end), 0) as number_of_open_roof_car
from {{ source('raw', 'train') }} t
left join {{ source('raw', 'cars') }} c on t.id = c.train_id
group by t.id, t.direction
