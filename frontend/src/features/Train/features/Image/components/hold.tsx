const TestTransformsDnd = ({
    control,
    errors,
    invenName,
  }: {
    control: Control<ParameterData, unknown>;
    errors: FieldErrors<ParameterData>;
    invenName: string;
  }) => {
    const { fields, move, insert, remove } = useFieldArray({
      control: control,
      name: "testTransforms",
    });
    const genTransformInvIds = () =>
      Object.fromEntries(
        STEP_SETTINGS.PARAMETERS.transformValues.map((transformValue) => [
          transformValue,
          Math.floor(Math.random() * Date.now()),
        ])
      );
    const [transformInvIds, setTransformInvIds] = useState<{
      [transformValue: string]: number;
    }>(genTransformInvIds());
    const [dndActive, setDndActive] = useState<Active | null>(null);
    const [invHovering, setInvHovering] = useState<boolean>(false);
  
    const dndActiveItem = useMemo(() => {
      if (!dndActive) return;
      if (dndActive.data.current && "inventory" in dndActive.data.current) {
        const value = dndActive.data.current.inventory
          .value as (typeof STEP_SETTINGS.PARAMETERS.transformValues)[number];
        return {
          id: transformInvIds[value],
          value: value,
          parameters: STEP_SETTINGS.PARAMETERS.transforms[value].parameters.map(
            () => ""
          ) as ""[],
        };
      } else if (dndActive.data.current && "sortable" in dndActive.data.current) {
        return fields[dndActive.data.current.sortable.index];
      }
    }, [dndActive]);
    const sensors = useSensors(
      useCustomPointerSensor(),
      useCustomKeyboardSensor({ coordinateGetter: sortableKeyboardCoordinates })
    );
    return (
      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragStart={({ active }) => {
          if (dndActive !== null) return;
          setDndActive(active);
        }}
        onDragOver={({ over }) => {
          if (!over || !over.data.current) {
            setInvHovering(false);
            return;
          }
          if (!invHovering) {
            setInvHovering(true);
          }
        }}
        onDragEnd={({ active, over }) => {
          if (dndActive && dndActive.data.current && dndActiveItem) {
            if (
              "inventory" in dndActive.data.current &&
              over?.data.current &&
              "sortable" in over.data.current
            ) {
              insert(over.data.current.sortable.index, {
                value: dndActiveItem.value,
                parameters: dndActiveItem.parameters as number[],
              });
            } else if (
              "sortable" in dndActive.data.current &&
              over?.data.current &&
              "sortable" in over.data.current
            ) {
              move(
                fields.findIndex((field) => field.id === active.id),
                fields.findIndex((field) => field.id === over.id)
              );
            }
          }
          setTransformInvIds(genTransformInvIds());
          setInvHovering(false);
          setDndActive(null);
        }}
        onDragCancel={({ active }) => {
          if (active.data.current && "inventory" in active.data.current) {
            const index = fields.findIndex((field) => field.id === active.id);
            if (index !== -1) {
              remove(fields.findIndex((field) => field.id === active.id));
            }
          }
          setTransformInvIds(genTransformInvIds());
          setInvHovering(false);
          setDndActive(null);
        }}
      >
        <Paper elevation={1} style={{ backgroundColor: "transparent" }}>
          <Stack alignItems={"center"} spacing={2} padding={2}>
            <Typography variant="h2" fontSize={25}>
            {invenName}
            </Typography>
            <Stack direction={"row"} spacing={2} justifyContent={"center"} sx={{ flexWrap: 'wrap', gap: 1 }}>
            {STEP_SETTINGS.PARAMETERS.transformValues.map((value) => (
              <TransformInventoryComponent
              id={transformInvIds[value]}
              key={value}
              value={value}
              />
            ))}
            </Stack>
          </Stack>
        </Paper>
        <Container>
          <Stack spacing={0}>
            <SortableContext
              items={
                dndActiveItem &&
                dndActive?.data.current &&
                "inventory" in dndActive.data.current
                  ? [dndActiveItem, ...fields]
                  : fields
              }
              strategy={verticalListSortingStrategy}
            >
              {fields.length > 0 ? (
                [
                  dndActiveItem &&
                  dndActive?.data.current &&
                  "inventory" in dndActive.data.current &&
                  invHovering ? (
                    <TransformComponent
                      key={dndActiveItem.id}
                      id={dndActiveItem.id}
                      data={dndActiveItem as ParameterData["trainTransforms"][number]}
                    />
                  ) : null,
                  ...fields.map((field, index) => (
                    <TransformComponent
                      key={field.id}
                      id={field.id}
                      data={field}
                      formProps={{
                        index: index,
                        control: control,
                        errors: errors,
                        remove: () => remove(index),
                      }}
                    />
                  )),
                ]
              ) : (
                <Card>This is Unimplemented</Card>
              )}
            </SortableContext>
          </Stack>
        </Container>
  
        <ClientOnlyPortal selector="#portal">
          <DragOverlay style={{ width: undefined }}>
            {dndActiveItem ? (
              dndActive?.data.current && "sortable" in dndActive.data.current ? (
                <TransformComponent
                  data={dndActiveItem as ParameterData["trainTransforms"][number]}
                  formProps={{
                    index: dndActive.data.current.sortable.index,
                    control: control,
                    errors: errors,
                  }}
                />
              ) : (
                <TransformComponent
                  data={dndActiveItem as ParameterData["trainTransforms"][number]}
                />
              )
            ) : null}
          </DragOverlay>
        </ClientOnlyPortal>
      </DndContext>
    );
  };
  
  const TrainTransformComponent = ({
    id,
    data,
    formProps,
  }: {
    id?: string | number;
    data: ParameterData["trainTransforms"][number];
    formProps?: {
      index: number;
      control: Control<ParameterData, unknown>;
      errors: FieldErrors<ParameterData>;
      remove?: () => void;
    };
  }) => {
    const {
      attributes,
      listeners,
      setNodeRef,
      isDragging,
      transform,
      transition,
    } = id
      ? useSortable({ id })
      : {
          attributes: undefined,
          listeners: undefined,
          setNodeRef: undefined,
          isDragging: undefined,
          transform: undefined,
          transition: undefined,
        };
    const style = transform
      ? {
          opacity: isDragging ? 0.4 : undefined,
          transform: CSS.Transform.toString(transform),
          transition: transition,
        }
      : undefined;
    return (
      <div ref={setNodeRef}>
        <Card
          sx={{ p: 3 }}
          style={{ ...{ display: "inline-block" }, ...style }}
          {...attributes}
          {...listeners}
        >
          <Stack
            direction={"row"}
            justifyContent={"space-between"}
            alignItems={"center"}
            spacing={3}
          >
            <Typography variant="h3" fontSize={18}>
              {STEP_SETTINGS.PARAMETERS.transforms[data.value].label}
            </Typography>
            <Stack direction={"row"} alignItems={"center"} spacing={3}>
              <Stack
                direction={"row"}
                alignItems={"center"}
                justifyContent={"flex-end"}
                spacing={2}
                divider={<Divider orientation="vertical" flexItem />}
              >
                {STEP_SETTINGS.PARAMETERS.transforms[data.value].parameters.map(
                  (parameter, index) => (
                    <div key={index} data-no-dnd>
                      {formProps ? (
                        <Controller
                          name={`trainTransforms.${formProps.index}.parameters.${index}`}
                          control={formProps.control}
                          rules={{ required: true }}
                          render={({ field: { onChange, value } }) => (
                            <TextField
                              label={parameter.label}
                              size={"small"}
                              type={parameter.type}
                              onChange={onChange}
                              value={value}
                              required
                              error={
                                formProps.errors.trainTransforms?.[formProps.index]
                                  ?.parameters?.[index]
                                  ? true
                                  : false
                              }
                            />
                          )}
                        />
                      ) : (
                        <TextField
                          label={parameter.label}
                          size={"small"}
                          type={parameter.type}
                          value={""}
                          required
                        />
                      )}
                    </div>
                  )
                )}
              </Stack>
              <div data-no-dnd>
                <IconButton onClick={formProps?.remove}>
                  <DeleteIcon />
                </IconButton>
              </div>
            </Stack>
          </Stack>
        </Card>
      </div>
    );
  };
  
  const TrainTransformInventoryComponent = ({
    id,
    value,
  }: {
    id: number;
    value: (typeof STEP_SETTINGS.PARAMETERS.transformValues)[number];
  }) => {
    const { attributes, listeners, isDragging, setNodeRef } = useDraggable({
      id: id,
      data: {
        inventory: {
          value,
        },
      },
    });
  
    const style = {
      opacity: isDragging ? 0.4 : undefined,
    };
    return (
      <div ref={setNodeRef}>
        <Card
          sx={{ p: 1 }}
          style={{ ...{ display: "inline-block" }, ...style }}
          {...attributes}
          {...listeners}
        >
          {STEP_SETTINGS.PARAMETERS.transforms[value].label}
        </Card>
      </div>
    );
  };
  